#!/usr/bin/env python3
"""
Download ALL AWS Braket tasks in a time window (no matching), including FAILED.

- Input times accept 'Aug 27, 2025 15:06 (UTC)', ISO, or epoch.
- Queries search_quantum_tasks per status, hydrates with get_quantum_task.
- For each task:
    <dir_root>/aws_all/<YYYYmmddTHHMMSSZ>__<STATUS>__<REGION>__shots<SHOTS>__<arnTail>/
        - (downloaded S3 payload if any)
        - aws_task_detail.json
        - task_status.txt (COMPLETED/FAILED/CANCELLED/UNKNOWN)
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError


# ---------------- Time parsing ----------------

_HUMAN_UTC_RE = re.compile(
    r"^\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+\d{2}:\d{2}(?::\d{2})?)\s*(?:\(UTC\)|UTC)?\s*$"
)

def _to_utc_datetime(t: Any) -> datetime:
    """Accept epoch, ISO, or 'Aug 27, 2025 15:06 (UTC)'; return aware UTC datetime."""
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(int(t), tz=timezone.utc)
    if isinstance(t, str):
        s = t.strip()
        m = _HUMAN_UTC_RE.match(s)
        if m:
            base = m.group(1)
            for fmt in ("%b %d, %Y %H:%M:%S", "%b %d, %Y %H:%M"):
                try:
                    dt = datetime.strptime(base, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
        s_iso = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s_iso)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.fromtimestamp(int(s), tz=timezone.utc)
    if isinstance(t, datetime):
        return t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
    raise TypeError(f"Unsupported time type: {type(t)}")

def _iso_utc(dt: datetime) -> str:
    """RFC3339 Zulu without microseconds (e.g., '2025-08-27T15:06:00Z')."""
    dtu = _to_utc_datetime(dt)
    return dtu.strftime("%Y-%m-%dT%H:%M:%SZ")

def _epoch_seconds(dt: datetime) -> int:
    return int(_to_utc_datetime(dt).timestamp())


# ---------------- Braket querying (search + hydrate) ----------------

def _search_one_status(client, v0: str, v1: str, status: str, max_results_per_page: int = 50):
    """Yield summaries for a single status using correct filters (EQUAL, not EQUALS)."""
    filters = [
        {"name": "createdAt", "operator": "BETWEEN", "values": [v0, v1]},
        {"name": "status",    "operator": "EQUAL",   "values": [status]},
    ]
    next_token: Optional[str] = None
    while True:
        kwargs = {"filters": filters, "maxResults": max_results_per_page}
        if next_token:  # only include when present
            kwargs["nextToken"] = next_token
        resp = client.search_quantum_tasks(**kwargs)
        for s in resp.get("quantumTasks", []):
            yield s
        next_token = resp.get("nextToken")
        if not next_token:
            break

def list_all_braket_tasks(
    time0: Any,
    time1: Any,
    statuses: Tuple[str, ...] = ("COMPLETED", "FAILED"),
    regions: Optional[List[str]] = None,
    max_results_per_page: int = 50,
) -> List[Dict[str, Any]]:
    """
    Use search_quantum_tasks with:
      createdAt BETWEEN [time0, time1] AND status EQUAL <status>   (loop per status)
    Then get_quantum_task per ARN for shots/S3/metadata.
    Returns tasks sorted by createdAt ascending.
    """
    t0 = _to_utc_datetime(time0)
    t1 = _to_utc_datetime(time1)
    if t1 < t0:
        raise ValueError("time1 must be >= time0")
    v0, v1 = _iso_utc(t0), _iso_utc(t1)

    session = boto3.session.Session()
    if regions is None:
        regions = session.get_available_regions("braket")
        if not regions:
            raise RuntimeError("No Braket regions available for 'braket'.")

    tasks: List[Dict[str, Any]] = []
    for region in regions:
        client = session.client("braket", region_name=region)
        for status in statuses:
            try:
                for s in _search_one_status(client, v0, v1, status, max_results_per_page):
                    arn = s.get("quantumTaskArn")
                    try:
                        detail = client.get_quantum_task(quantumTaskArn=arn)
                    except (ClientError, BotoCoreError) as e:
                        detail = {"_get_error": str(e), "quantumTaskArn": arn}

                    created_at = s.get("createdAt") or detail.get("createdAt")
                    ended_at   = s.get("endedAt")   or detail.get("endedAt")
                    if isinstance(created_at, str): created_at = _to_utc_datetime(created_at)
                    if isinstance(ended_at, str):   ended_at   = _to_utc_datetime(ended_at)

                    item = {
                        "region": region,
                        "quantumTaskArn": arn,
                        "status": s.get("status") or detail.get("status"),
                        "deviceArn": s.get("deviceArn") or detail.get("deviceArn"),
                        "createdAt": created_at,
                        "endedAt": ended_at,
                        "shots": detail.get("shots"),
                        "billableShots": detail.get("billableShots"),
                        "outputS3Bucket": detail.get("outputS3Bucket"),
                        "outputS3Directory": detail.get("outputS3Directory"),
                        "resultS3Bucket": detail.get("resultS3Bucket"),
                        "resultS3Directory": detail.get("resultS3Directory"),
                        "additionalMetadata": detail.get("additionalMetadata"),
                        "_summary": s,
                    }
                    qmeta = (item.get("additionalMetadata") or {}).get("queraMetadata", {})
                    item["numSuccessfulShots"] = qmeta.get("numSuccessfulShots")
                    tasks.append(item)
            except (ClientError, BotoCoreError) as e:
                print(f"[WARN] search_quantum_tasks error in {region}: {e}")

    tasks.sort(key=lambda x: _epoch_seconds(x["createdAt"]) if x.get("createdAt") else 0)
    return tasks


# ---------------- S3 download helpers ----------------

def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        cand = Path(str(path) + f"-{i}")
        if not cand.exists():
            return cand
        i += 1

def _safe(s: Optional[str]) -> str:
    if not s:
        return "NA"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _folder_name_for_task(t: Dict[str, Any]) -> str:
    ts = t.get("createdAt")
    ts_str = _to_utc_datetime(ts).strftime("%Y%m%dT%H%M%SZ") if ts else "noTime"
    status = t.get("status") or "UNKNOWN"
    region = t.get("region") or "NA"
    shots = t.get("shots")
    shots_str = f"shots{shots}" if shots is not None else "shotsNA"
    arn = t.get("quantumTaskArn") or "arn"
    arn_tail = arn.split("/")[-1]
    return f"{ts_str}__{status}__{region}__{shots_str}__{_safe(arn_tail)}"

def _s3_download_prefix(s3, bucket: Optional[str], prefix: Optional[str], dest_dir: Path) -> int:
    if not bucket or not prefix:
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    n_files = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else key
            outpath = dest_dir / rel
            outpath.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(outpath))
            n_files += 1
    return n_files

def download_one_task(t: Dict[str, Any], dir_root: str, base_folder: str = "aws_all") -> Path:
    """
    Create a folder for the task, download its payload (if any), and write status + metadata.
    """
    base = Path(dir_root) / base_folder
    base.mkdir(parents=True, exist_ok=True)
    dest = _ensure_unique_dir(base / _folder_name_for_task(t))

    # Prefer resultS3*, fall back to outputS3*
    bucket = t.get("resultS3Bucket") or t.get("outputS3Bucket")
    prefix = t.get("resultS3Directory") or t.get("outputS3Directory")

    s3 = boto3.client("s3")
    downloaded = 0
    try:
        downloaded = _s3_download_prefix(s3, bucket, prefix, dest)
    except (ClientError, BotoCoreError) as e:
        print(f"[WARN] S3 download failed for {t.get('quantumTaskArn')}: {e}")

    # Sidecars
    with open(dest / "aws_task_detail.json", "w", encoding="utf-8") as f:
        json.dump(t, f, indent=2, default=str)

    status = t.get("status") or "UNKNOWN"
    (dest / "task_status.txt").write_text(f"{status}\n", encoding="utf-8")

    print(f"[INFO] {status} | files={downloaded} -> {dest}")
    return dest


# ---------------- main ----------------

if __name__ == "__main__":
    # ---- Configure here ----
    time0 = "Aug 26, 2025 19:02 (UTC)" # start time
    time1 = "Aug 26, 2025 19:05 (UTC)" # end time

    dir_root = "diagnose_dir"             # your experiment root
    regions = None                         # or e.g., ["us-east-1","us-west-2","eu-west-2"]
    STATUSES = ("COMPLETED", "FAILED")     # add "CANCELLED" if you want those too

    print("Listing ALL AWS Braket tasks in window...")
    aws_tasks = list_all_braket_tasks(time0, time1, statuses=STATUSES, regions=regions)
    print(f"Found {len(aws_tasks)} tasks.")

    for t in aws_tasks:
        download_one_task(t, dir_root=dir_root, base_folder="aws_all")
