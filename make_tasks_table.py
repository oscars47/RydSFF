#!/usr/bin/env python3
"""
List AWS Braket tasks in a time window and emit a LaTeX table (no downloads).

- Input times accept 'Aug 27, 2025 15:06 (UTC)', ISO, or epoch.
- Queries search_quantum_tasks per status, hydrates with get_quantum_task.
- Produces a LaTeX table with a caption showing totals by status.
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
        if next_token:
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
    Use search_quantum_tasks with createdAt BETWEEN [time0, time1] AND status EQUAL <status>;
    then get_quantum_task per ARN for details. Return sorted by createdAt ascending.
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
                        "additionalMetadata": detail.get("additionalMetadata"),
                    }
                    qmeta = (item.get("additionalMetadata") or {}).get("queraMetadata", {})
                    item["numSuccessfulShots"] = qmeta.get("numSuccessfulShots")
                    tasks.append(item)
            except (ClientError, BotoCoreError) as e:
                print(f"[WARN] search_quantum_tasks error in {region}: {e}")

    tasks.sort(key=lambda x: _epoch_seconds(x["createdAt"]) if x.get("createdAt") else 0)
    return tasks

# ---------------- LaTeX helpers ----------------

_LATEX_REPL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}

def latex_escape(x: Any) -> str:
    if x is None:
        return "--"
    s = str(x)
    return "".join(_LATEX_REPL.get(ch, ch) for ch in s)

def arn_tail(arn: Optional[str]) -> str:
    if not arn:
        return "--"
    return arn.split("/")[-1]

def device_short(device_arn: Optional[str]) -> str:
    if not device_arn:
        return "--"
    # Keep final component (device name) for compactness
    return device_arn.split("/")[-1]

def fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return "--"
    return _to_utc_datetime(dt).strftime("%Y-%m-%d %H:%M:%SZ")

def duration_seconds(created: Optional[datetime], ended: Optional[datetime]) -> Optional[int]:
    if not created or not ended:
        return None
    return int((_to_utc_datetime(ended) - _to_utc_datetime(created)).total_seconds())

def build_latex_table(tasks: List[Dict[str, Any]], caption_note: str, label: str = "tab:braket_tasks") -> str:
    """
    Build a LaTeX table (booktabs) summarizing tasks.
    Columns: Created (UTC), Ended (UTC), Task Name, Status
    """
    n_completed = sum(1 for t in tasks if (t.get("status") or "").upper() == "COMPLETED")
    n_failed    = sum(1 for t in tasks if (t.get("status") or "").upper() == "FAILED")

    caption = (
        f"{latex_escape(caption_note)} "
        f"(Completed: {n_completed}, Failed: {n_failed}; Total: {len(tasks)})."
    )

    header = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\begin{tabular}{@{}llll@{}}" "\n"
        r"\toprule" "\n"
        r"Created (UTC) & Ended (UTC) & Task Name & Status \\" "\n"
        r"\midrule" "\n"
    )

    rows = []
    for t in tasks:
        created = fmt_dt(t.get("createdAt"))
        ended   = fmt_dt(t.get("endedAt"))
        tail    = arn_tail(t.get("quantumTaskArn"))
        status  = t.get("status") or "UNKNOWN"

        row = " & ".join([
            latex_escape(created),
            latex_escape(ended),
            latex_escape(tail),
            latex_escape(status),
        ]) + r" \\"
        rows.append(row)

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        rf"\caption{{{caption}}}" "\n"
        rf"\label{{{label}}}" "\n"
        r"\end{table}" "\n"
    )

    return header + "\n".join(rows) + "\n" + footer


if __name__ == "__main__":
    # ---- Configure here ----
    time0 = "Aug 15, 2025 00:00 (UTC)"  # start time
    time1 = "Sep 1, 2025 19:05 (UTC)"  # end time

    regions = None
    STATUSES = ("COMPLETED", "FAILED")

    print("Listing AWS Braket tasks in window...")
    tasks = list_all_braket_tasks(time0, time1, statuses=STATUSES, regions=regions)
    print(f"Found {len(tasks)} tasks.")

    caption_note = f"Braket tasks between { _iso_utc(_to_utc_datetime(time0)) } and { _iso_utc(_to_utc_datetime(time1)) } (UTC)"
    latex_table = build_latex_table(tasks, caption_note, label="tab:aws_braket_tasks")

    out_path = Path("aws_tasks_table.tex")
    out_path.write_text(latex_table, encoding="utf-8")
    print(f"Wrote LaTeX table -> {out_path.resolve()}")
    print("\n===== LaTeX table begin =====\n")
    print(latex_table)
    print("\n===== LaTeX table end =====\n")

