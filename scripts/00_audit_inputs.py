#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit expected input files for the anatomical lung resection phase-recognition pipeline.

This script intentionally checks only file presence and schema-like fields. It does not
export patient-level contents.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--frame-root", type=Path, required=False)
    p.add_argument("--metadata-csv", type=Path, required=False)
    p.add_argument("--case-pattern", type=str, default="CASE*.json")
    p.add_argument("--out-csv", type=Path, default=Path("outputs/input_audit.csv"))
    return p.parse_args()


def case_id_from_name(name: str) -> str:
    m = re.search(r"(CASE\d{3})", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else name


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.annotation_root.rglob(args.case_pattern)):
        case_id = case_id_from_name(path.name)
        ok_json = False
        n_segments = None
        labels = []
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            ok_json = True
            text = json.dumps(obj, ensure_ascii=False)
            labels = sorted(set(re.findall(r'"label"\s*:\s*"([^"]+)"', text)))
            n_segments = text.count('"label"')
        except Exception:
            pass
        rows.append({
            "case_id": case_id,
            "annotation_json": str(path),
            "json_readable": ok_json,
            "n_label_entries_approx": n_segments,
            "labels_seen": ";".join(labels),
        })
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
