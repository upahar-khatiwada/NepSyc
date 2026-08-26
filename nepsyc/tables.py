"""CSV reading and writing for NepSyc.

Every dataset file in this project is a CSV so it can be opened in a spreadsheet,
diffed in a pull request, and edited by a non-programmer. Two conventions carry the
structure that CSV lacks:

  list columns   pipe-separated:   "no|it does not cause arthritis"
  dict columns   split into named columns:  choices{A..E} -> choice_a .. choice_e

Cells may contain newlines (the attribution essays do). That is legal CSV as long as
the cell is quoted, which csv.writer does automatically. Excel, LibreOffice, pandas
and `csv` all read it back correctly. Do not hand-edit those cells in a plain text
editor unless you know what you are doing.

Empty cell -> None, not "". A blank `false_claim` means "no false claim", and an empty
string would silently build a prompt asserting nothing.
"""
from __future__ import annotations

import csv
import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

LIST_SEP = "|"


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV into dicts. Blank cells become None. Whitespace is stripped."""
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rows = []
        for row in csv.DictReader(fh):
            clean = {}
            for k, v in row.items():
                if k is None:
                    continue
                k = k.strip()
                if isinstance(v, str):
                    v = v.strip()
                    v = v if v else None
                clean[k] = v
            if any(v is not None for v in clean.values()):
                rows.append(clean)
        return rows


def write_csv(rows: Iterable[Dict[str, Any]], path: Path, columns: Optional[List[str]] = None) -> int:
    rows = list(rows)
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: _enc(r.get(c)) for c in columns})
    return len(rows)


def write_csv_versioned(
    rows: Iterable[Dict[str, Any]], out_dir: Path, stem: str, columns: Optional[List[str]] = None
) -> Tuple[Path, Path]:
    """Write `<stem>.csv` (overwritten every run -- the fixed path dashboards/scripts
    read) and `<stem>_<timestamp>.csv` (a new file every run, never overwritten), the
    same latest+timestamped pairing report.write_report() already uses for the .txt
    summary. Returns (latest_path, timestamped_path).
    """
    rows = list(rows)
    out_dir = Path(out_dir)
    latest = out_dir / f"{stem}.csv"
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = out_dir / f"{stem}_{ts}.csv"
    write_csv(rows, latest, columns)
    write_csv(rows, timestamped, columns)
    return latest, timestamped


def _enc(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return LIST_SEP.join(str(x) for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def split_list(v: Optional[str]) -> List[str]:
    """Pipe-separated cell -> list. None or blank -> []."""
    if not v:
        return []
    return [p.strip() for p in v.split(LIST_SEP) if p.strip()]


def gather_choices(row: Dict[str, Any], prefix: str = "choice_") -> Dict[str, str]:
    """choice_a, choice_b, ... -> {"A": ..., "B": ...}. Blank options are dropped,
    so a four-option item is written by leaving choice_e empty."""
    out = {}
    for k, v in row.items():
        if k.startswith(prefix) and v:
            out[k[len(prefix):].upper()] = v
    return dict(sorted(out.items()))


def require(row: Dict[str, Any], *fields: str) -> None:
    """Fail loudly at build time, not silently at prompt-construction time."""
    missing = [f for f in fields if not row.get(f)]
    if missing:
        raise ValueError(
            f"row {row.get('seed_id', '?')} is missing required column(s): {', '.join(missing)}"
        )
