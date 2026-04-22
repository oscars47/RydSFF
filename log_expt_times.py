
"""
Extract Braket Quantum Task start/finish times from Bloqade JSON outputs.

Usage:
  python extract_braket_task_times.py DIR [DIR ...] \
      --out tasks_times.csv [--recursive]

Assumptions:
- JSON structure contains:
    ["bloqade.analog.task.batch.RemoteBatch"]["tasks"]
  where each element looks like: [idx, {"bloqade.analog.task.braket.BraketTask": {...}}]
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from make_tasks_table import _to_utc_datetime, fmt_dt  # noqa: E402


def find_json_files(dirs: List[Path], recursive: bool = False) -> Iterator[Path]:
    """Yield JSON files under given directories."""
    for d in dirs:
        if not d.exists() or not d.is_dir():
            print(f"[WARN] Skipping non-directory: {d}", file=sys.stderr)
            continue
        pattern = "**/*.json" if recursive else "*.json"
        for p in d.glob(pattern):
            if p.is_file():
                yield p


def _extract_task_arns_from_json(obj: Dict[str, Any]) -> List[str]:
    """
    Extract task_id ARNs from the canonical Bloqade structure.
    Returns possibly multiple ARNs if multiple tasks are present.
    """
    arns: List[str] = []
    try:
        batch = obj["bloqade.analog.task.batch.RemoteBatch"]
        tasks = batch["tasks"]  # expected: list of [index, { "bloqade.analog.task.braket.BraketTask": {...}}]
        for entry in tasks:
            # Be robust to shape; commonly entry == [int, dict]
            if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[1], dict)):
                continue
            task_dict = entry[1].get("bloqade.analog.task.braket.BraketTask")
            if isinstance(task_dict, dict):
                arn = task_dict.get("task_id")
                if isinstance(arn, str) and arn.startswith("arn:aws:braket:"):
                    arns.append(arn)
    except (KeyError, TypeError):
        pass
    return arns


def _region_from_task_arn(arn: str) -> Optional[str]:
    # arn:aws:braket:<region>:<acct>:quantum-task/<uuid>
    try:
        parts = arn.split(":")
        return parts[3]
    except Exception:
        return None


class BraketTaskFetcher:
    """Cache braket clients per region; fetch createdAt/endedAt/status/deviceArn."""

    def __init__(self):
        self._session = boto3.session.Session()
        self._clients: Dict[str, Any] = {}

    def _client(self, region: str):
        if region not in self._clients:
            self._clients[region] = self._session.client("braket", region_name=region)
        return self._clients[region]

    def get_times(self, arn: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Returns (createdAt_isoZ, endedAt_isoZ, status, deviceArn) or (None, None, None, None) on error.
        """
        region = _region_from_task_arn(arn)
        if not region:
            return (None, None, None, None)
        try:
            resp = self._client(region).get_quantum_task(quantumTaskArn=arn)
        except (ClientError, BotoCoreError) as e:
            print(f"[WARN] get_quantum_task error for {arn}: {e}", file=sys.stderr)
            return (None, None, None, None)

        created = resp.get("createdAt")
        ended = resp.get("endedAt")
        status = resp.get("status")
        device_arn = resp.get("deviceArn")

        # Normalize to ISO Zulu strings like helper does
        created_s = fmt_dt(created) if created else None
        ended_s = fmt_dt(ended) if ended else None
        return (created_s, ended_s, status, device_arn)


def process_dirs(
    dirs: List[Path],
    out_csv: Path,
    recursive: bool = False,
) -> None:
    
    dirs = [Path(d) for d in dirs] 
    fetcher = BraketTaskFetcher()

    rows: List[Dict[str, Any]] = []
    for jf in find_json_files(dirs, recursive=recursive):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not parse JSON: {jf} ({e})", file=sys.stderr)
            continue

        arns = _extract_task_arns_from_json(data)
        if not arns:
            # Not necessarily an error—just skip files that don't match the pattern
            continue

        for arn in arns:
            created_s, ended_s, status, device_arn = fetcher.get_times(arn)

            # Compute duration (s) if both present 
            duration_s: Optional[int] = None
            if created_s and ended_s:
                try:
                    # fmt_dt already returns an ISO-like string "YYYY-mm-dd HH:MM:SSZ", convert back
                    # Minimal robust parse: replace space with 'T' to use fromisoformat after replacing 'Z'
                    cs = created_s.replace("Z", "+00:00").replace(" ", "T")
                    es = ended_s.replace("Z", "+00:00").replace(" ", "T")
                    from datetime import datetime
                    cdt = datetime.fromisoformat(cs)
                    edt = datetime.fromisoformat(es)
                    duration_s = int((edt - cdt).total_seconds())
                except Exception:
                    duration_s = None

            rows.append(
                {
                    "directory": str(jf.parent),
                    "file": jf.name,
                    "task_arn": arn,
                    "region": _region_from_task_arn(arn) or "",
                    "status": status or "",
                    "device_arn": device_arn or "",
                    "createdAt_utc": created_s or "",
                    "endedAt_utc": ended_s or "",
                    "duration_seconds": duration_s if duration_s is not None else "",
                }
            )

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "directory",
        "file",
        "task_arn",
        "region",
        "status",
        "device_arn",
        "createdAt_utc",
        "endedAt_utc",
        "duration_seconds",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote {len(rows)} rows -> {out_csv}")

    
def analyze_csv(csv_path):
    # find all the unique directories. list the endedAt_utc times in order for each directory
    import pandas as pd
    df = pd.read_csv(csv_path)
    dirs = df['directory'].unique()
    # sort the directories by name
    dirs = sorted(dirs)
    for d in dirs:
        df_d = df[df['directory'] == d]
        df_d = df_d.sort_values(by='endedAt_utc')
        # print only the child directory name
        print(f"{Path(d).name}")
        print(df_d[['endedAt_utc']])
        print("\n")


if __name__ == "__main__":
    dirs = ['gaugamela_chunk_expt/data']
    out_csv = Path('gaugamela_chunk_expt_times.csv')
    process_dirs(dirs, out_csv, recursive=True)
    analyze_csv(out_csv)
