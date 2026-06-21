#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute case-level workflow metrics and correlate them with operative outcomes.

Expected frame-level input columns:
  case_id, frame_index or time_sec, gt_name, pred_name

At least one of gt_name/pred_name is required. If time_sec is unavailable, each row is
assumed to represent one sampled frame and duration is estimated from --sample-fps.

Expected metadata input columns:
  case_id, operative_time_min
Optional:
  blood_loss_g, side, procedure, target
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-level-csv", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/case_workflow_outcome"))
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--case-col", type=str, default="case_id")
    p.add_argument("--operative-time-col", type=str, default="operative_time_min")
    p.add_argument("--blood-loss-col", type=str, default="blood_loss_g")
    p.add_argument("--labels", nargs="*", default=["inferior_hilum", "posterior_hilum", "superior_hilum", "anterior_hilum", "fissure", "uncertain"])
    return p.parse_args()


def _duration_per_row(df: pd.DataFrame, sample_fps: float) -> pd.Series:
    if "time_sec" in df.columns:
        # Use median positive delta within each case as a robust local frame duration.
        durs = []
        for _, g in df.groupby("case_id", sort=False):
            t = pd.to_numeric(g["time_sec"], errors="coerce")
            dt = t.diff().median()
            dur = float(dt) if pd.notna(dt) and dt > 0 else 1.0 / sample_fps
            durs.extend([dur] * len(g))
        return pd.Series(durs, index=df.index, dtype=float)
    return pd.Series(np.full(len(df), 1.0 / sample_fps), index=df.index, dtype=float)


def transition_count(labels: Iterable[str]) -> int:
    last = None
    n = 0
    for x in labels:
        if pd.isna(x):
            continue
        x = str(x)
        if last is None:
            last = x
            continue
        if x != last:
            n += 1
            last = x
    return n


def build_case_metrics(frame_df: pd.DataFrame, labels: list[str], source_col: str, prefix: str) -> pd.DataFrame:
    rows = []
    for case_id, g in frame_df.groupby("case_id", sort=True):
        total_sec = float(g["row_duration_sec"].sum())
        row = {
            "case_id": case_id,
            f"{prefix}_duration_min_est": total_sec / 60.0,
            f"{prefix}_transition_count": transition_count(g[source_col].tolist()),
        }
        for lab in labels:
            mask = g[source_col].astype(str) == lab
            sec = float(g.loc[mask, "row_duration_sec"].sum())
            row[f"{prefix}_duration_min_{lab}"] = sec / 60.0
            row[f"{prefix}_fraction_{lab}"] = sec / total_sec if total_sec > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    rows = []
    for col in df.columns:
        if col in {"case_id", outcome_col}:
            continue
        if not (col.startswith("pred_") or col.startswith("true_")):
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        ok = x.notna() & y.notna()
        if int(ok.sum()) < 4:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        rows.append({"metric": col, "outcome": outcome_col, "n": int(ok.sum()), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows).sort_values(["outcome", "spearman_rho"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.frame_level_csv)
    meta = pd.read_csv(args.metadata_csv)
    frame = frame.rename(columns={args.case_col: "case_id"})
    meta = meta.rename(columns={args.case_col: "case_id", args.operative_time_col: "operative_time_min"})
    if args.blood_loss_col in meta.columns:
        meta = meta.rename(columns={args.blood_loss_col: "blood_loss_g"})
    frame["row_duration_sec"] = _duration_per_row(frame, args.sample_fps)

    case_parts = []
    if "pred_name" in frame.columns:
        case_parts.append(build_case_metrics(frame, args.labels, "pred_name", "pred"))
    if "gt_name" in frame.columns:
        case_parts.append(build_case_metrics(frame, args.labels, "gt_name", "true"))
    if not case_parts:
        raise ValueError("frame-level CSV must contain pred_name and/or gt_name")

    case_metrics = case_parts[0]
    for part in case_parts[1:]:
        case_metrics = case_metrics.merge(part, on="case_id", how="outer")
    merged = case_metrics.merge(meta, on="case_id", how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "case_workflow_metrics_with_outcomes.csv", index=False, encoding="utf-8-sig")

    corr_tables = []
    if "operative_time_min" in merged.columns:
        corr_tables.append(correlation_table(merged, "operative_time_min"))
    if "blood_loss_g" in merged.columns:
        corr_tables.append(correlation_table(merged, "blood_loss_g"))
    if corr_tables:
        corr = pd.concat(corr_tables, ignore_index=True)
        corr.to_csv(args.out_dir / "workflow_outcome_correlations.csv", index=False, encoding="utf-8-sig")
        print(corr.head(20).to_string(index=False))
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
