#!/usr/bin/env python3
"""parse_rabi_reports_json.py

Recursively parse Markdown analysis reports and compile site-resolved parameters into JSON.

Only parses a report if its FIRST LINE exactly equals the required string.

Extracted per parsed report (one JSON object per report):
- timestamp from **Runtime:**
- site_ls: list of site indices (sorted)
- Omega_ls, Omega_unc_ls: angular frequencies in us^-1, computed as ω = 2π f
    MHz -> 2π * value        (us^-1)
    kHz -> 2π * value * 1e-3 (us^-1)
    Hz  -> 2π * value * 1e-6 (us^-1)
  (If unit missing, assumes already in us^-1 and leaves unchanged.)
- tau_ls, tau_unc_ls: times in us (ns -> 1e-3 us)
- epsilon_g_ls, epsilon_g_unc_ls: probabilities (m -> 1e-3)
- epsilon_r_ls, epsilon_r_unc_ls: probabilities (m -> 1e-3)

CLI:
  python parse_rabi_reports_json.py ROOT_DIR --out out.json
  python parse_rabi_reports_json.py ROOT_DIR --required-first-line "# ..." --out out.json
"""

from __future__ import annotations
import os, re, json, argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from datetime import datetime, timezone

DEFAULT_REQUIRED_FIRST_LINE = "# rabi_flop_omega_2.5mhz_highbz_v0 Analysis"
# DEFAULT_REQUIRED_FIRST_LINE = "# rabi_flop_omega_2.5mhz_gd_1mhz_ld_1mhz_v0 Analysis"

def _parse_number_with_unit(s: str) -> Tuple[Optional[float], Optional[str]]:
    if s is None:
        return None, None
    s = str(s).strip()
    if s.lower() in {"none", "nan", ""}:
        return None, None
    s = s.replace("−", "-").replace("µ", "u")
    s = s.lstrip("±").strip()
    m = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z/%_^0-9]+)?$", s)
    if not m:
        return None, None
    return float(m.group(1)), (m.group(2) if m.group(2) else None)

def angfreq_to_us_inv(x: str) -> Optional[float]:
    """Convert MHz/kHz/Hz -> angular frequency (ω=2πf) in us^-1."""
    val, unit = _parse_number_with_unit(x)
    if val is None:
        return None
    if unit is None:
        return val
    unit = unit.lower()
    if unit == "mhz":
        return 2*np.pi * val
    if unit == "khz":
        return 2*np.pi * val * 1e-3
    if unit == "hz":
        return 2*np.pi * val * 1e-6
    raise ValueError(f"Unrecognized frequency unit: {unit} in '{x}'")

def time_to_us(x: str) -> Optional[float]:
    val, unit = _parse_number_with_unit(x)
    if val is None:
        return None
    if unit is None:
        return val
    unit = unit.lower()
    if unit == "us":
        return val
    if unit == "ns":
        return val * 1e-3
    if unit == "ms":
        return val * 1e3
    if unit == "s":
        return val * 1e6
    raise ValueError(f"Unrecognized time unit: {unit} in '{x}'")

def prob_from_milli(x: str) -> Optional[float]:
    val, unit = _parse_number_with_unit(x)
    if val is None:
        return None
    if unit is None:
        return val
    unit = unit.lower()
    if unit == "m":
        return val * 1e-3
    if unit in {"%", "percent"}:
        return val / 100.0
    raise ValueError(f"Unrecognized probability unit: {unit} in '{x}'")

def parse_markdown_table(lines: List[str], start_idx: int) -> Tuple[List[Dict[str, str]], int]:
    header = [h.strip() for h in lines[start_idx].strip().strip("|").split("|")]
    i = start_idx + 1
    if i < len(lines) and re.match(r"^\s*\|\s*[-: ]+\|", lines[i]):
        i += 1
    rows: List[Dict[str, str]] = []
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line.strip().startswith("|"):
            break
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= len(header):
            rows.append({header[j]: parts[j] for j in range(len(header))})
        i += 1
    return rows, i

def extract_runtime_timestamp(md_text: str) -> Optional[float]:
    """Extract timestamp from markdown and convert to Unix timestamp (seconds since epoch)."""
    m = re.search(r"\*\*Runtime:\*\*\s*([0-9T:\.\+\-]+)", md_text)
    if not m:
        return None
    
    timestamp_str = m.group(1)
    try:
        # Parse ISO format timestamp and convert to Unix timestamp
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return None

def extract_section_table(md_lines: List[str], section_title: str) -> List[Dict[str, str]]:
    heading = f"## {section_title}".strip()
    for idx, line in enumerate(md_lines):
        if line.strip() == heading:
            j = idx + 1
            while j < len(md_lines) and not md_lines[j].lstrip().startswith("|"):
                j += 1
            if j >= len(md_lines):
                return []
            rows, _ = parse_markdown_table(md_lines, j)
            return rows
    return []

def _rows_to_site_ordered_lists(rows: List[Dict[str, str]], value_fn) -> Tuple[List[int], List[Optional[float]], List[Optional[float]]]:
    parsed = []
    for r in rows:
        s = int(r["Site"])
        v = value_fn(r["Value"])
        u = value_fn(r["Uncertainty"])
        parsed.append((s, v, u))
    parsed.sort(key=lambda t: t[0])
    return [t[0] for t in parsed], [t[1] for t in parsed], [t[2] for t in parsed]

def parse_rabi_md_file_to_json(path: str, required_first_line: str) -> Optional[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines or lines[0].strip() != required_first_line:
        return None

    out: Dict[str, Any] = {
        "timestamp": extract_runtime_timestamp(text),
        "source_md": str(path),
    }

    freq_rows = extract_section_table(lines, "Site Resolved Extracted Frequency")
    site_ls, Omega_ls, Omega_unc_ls = _rows_to_site_ordered_lists(freq_rows, angfreq_to_us_inv)
    out.update({"site_ls": site_ls, "Omega_ls": Omega_ls, "Omega_unc_ls": Omega_unc_ls})

    tau_rows = extract_section_table(lines, "Site Resolved Extracted Decay Constant")
    site_ls2, tau_ls, tau_unc_ls = _rows_to_site_ordered_lists(tau_rows, time_to_us)
    out.update({"tau_ls": tau_ls, "tau_unc_ls": tau_unc_ls})
    if site_ls2 and site_ls2 != site_ls:
        out["site_ls_tau"] = site_ls2

    eg_rows = extract_section_table(lines, "Site Resolved Ground State Detection Error")
    site_ls3, eg_ls, eg_unc_ls = _rows_to_site_ordered_lists(eg_rows, prob_from_milli)
    out.update({"epsilon_g_ls": eg_ls, "epsilon_g_unc_ls": eg_unc_ls})
    if site_ls3 and site_ls3 != site_ls:
        out["site_ls_epsilon_g"] = site_ls3

    er_rows = extract_section_table(lines, "Site Resolved Excited State Detection Error")
    site_ls4, er_ls, er_unc_ls = _rows_to_site_ordered_lists(er_rows, prob_from_milli)
    out.update({"epsilon_r_ls": er_ls, "epsilon_r_unc_ls": er_unc_ls})
    if site_ls4 and site_ls4 != site_ls:
        out["site_ls_epsilon_r"] = site_ls4

    return out

def build_json_from_root(majd_root: str, required_first_line: str) -> Dict[str, Any]:
    all_timestamps = []
    all_dirpaths = []
    all_Omega_ls = []
    all_Omega_unc_ls = []
    all_tau_ls = []
    all_tau_unc_ls = []
    all_epsilon_g_ls = []
    all_epsilon_g_unc_ls = []
    all_epsilon_r_ls = []
    all_epsilon_r_unc_ls = []
    
    for dirpath, _, filenames in os.walk(majd_root):
        for fn in filenames:
            if fn.lower().endswith(".md"):
                p = os.path.join(dirpath, fn)
                try:
                    rec = parse_rabi_md_file_to_json(p, required_first_line=required_first_line)
                    if rec is not None:
                        all_timestamps.append(rec.get("timestamp"))
                        all_dirpaths.append(dirpath)
                        all_Omega_ls.append(rec.get("Omega_ls", []))
                        all_Omega_unc_ls.append(rec.get("Omega_unc_ls", []))
                        all_tau_ls.append(rec.get("tau_ls", []))
                        all_tau_unc_ls.append(rec.get("tau_unc_ls", []))
                        all_epsilon_g_ls.append(rec.get("epsilon_g_ls", []))
                        all_epsilon_g_unc_ls.append(rec.get("epsilon_g_unc_ls", []))
                        all_epsilon_r_ls.append(rec.get("epsilon_r_ls", []))
                        all_epsilon_r_unc_ls.append(rec.get("epsilon_r_unc_ls", []))
                except Exception:
                    pass
    
    return {
        "timestamp": all_timestamps,
        "dirpath": all_dirpaths,
        "Omega_ls": all_Omega_ls,
        "Omega_unc_ls": all_Omega_unc_ls,
        "tau_ls": all_tau_ls,
        "tau_unc_ls": all_tau_unc_ls,
        "epsilon_g_ls": all_epsilon_g_ls,
        "epsilon_g_unc_ls": all_epsilon_g_unc_ls,
        "epsilon_r_ls": all_epsilon_r_ls,
        "epsilon_r_unc_ls": all_epsilon_r_unc_ls,
    }

def parse_majd(majd_root: str, output_path: Optional[str] = "majd_calibs.json", required_first_line: str = DEFAULT_REQUIRED_FIRST_LINE):
    """Parse Markdown reports from majd_root and compile into JSON.
    
    Args:
        majd_root: Directory to recursively search for .md reports
        output_path: Optional output JSON path
        required_first_line: Exact first-line string required to parse a report
    """
    records = build_json_from_root(majd_root, required_first_line=required_first_line)
    text = json.dumps(records, indent=2, sort_keys=False)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
    print(text)

