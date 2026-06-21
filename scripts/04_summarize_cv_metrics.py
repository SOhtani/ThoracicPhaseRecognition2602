#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_existing_workflow_eval_results_v2.py

Reads existing workflow-classification evaluation results under:
  <project_root>/outputs/level{0,1,2}/_summary

Important:
  confusion_matrix.csv files are headerless matrices in this project.
  Therefore they must be read with header=None.

Outputs:
  - workflow_eval_per_fold_summary.csv
  - workflow_eval_pooled_confusion_summary.csv
  - workflow_eval_class_metrics.csv
  - workflow_eval_abstract_ready_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_OUT_DIR = Path("outputs/existing_eval_summary_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def level_dir_name(level: int) -> str:
    return f"level{level}"


def level_name(level: int) -> str:
    return f"Level-{level}"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def metrics_from_confusion(cm: np.ndarray, level: str) -> tuple[dict, pd.DataFrame]:
    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    diag = np.diag(cm)

    support = cm.sum(axis=1)
    predicted_n = cm.sum(axis=0)

    recall = np.divide(diag, support, out=np.full_like(diag, np.nan, dtype=float), where=support != 0)
    precision = np.divide(diag, predicted_n, out=np.full_like(diag, np.nan, dtype=float), where=predicted_n != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full_like(diag, np.nan, dtype=float),
        where=np.isfinite(precision + recall) & ((precision + recall) != 0),
    )

    supported = support > 0

    overall = {
        "level": level,
        "n_frames_pooled_confusion": int(total),
        "n_classes": int(cm.shape[0]),
        "accuracy_pooled_confusion": safe_div(float(diag.sum()), total),
        "macro_precision_pooled_confusion": float(np.nanmean(precision[supported])),
        "macro_recall_pooled_confusion": float(np.nanmean(recall[supported])),
        "macro_f1_pooled_confusion": float(np.nanmean(f1[supported])),
        "recall_min_pooled_confusion": float(np.nanmin(recall[supported])),
        "recall_max_pooled_confusion": float(np.nanmax(recall[supported])),
        "recall_mean_pooled_confusion": float(np.nanmean(recall[supported])),
    }

    class_rows: List[dict] = []
    for i in range(cm.shape[0]):
        class_rows.append(
            {
                "level": level,
                "class_index": i,
                "support": int(support[i]),
                "predicted_n": int(predicted_n[i]),
                "tp": int(diag[i]),
                "fp": int(predicted_n[i] - diag[i]),
                "fn": int(support[i] - diag[i]),
                "precision": float(precision[i]) if np.isfinite(precision[i]) else np.nan,
                "recall": float(recall[i]) if np.isfinite(recall[i]) else np.nan,
                "f1": float(f1[i]) if np.isfinite(f1[i]) else np.nan,
            }
        )

    return overall, pd.DataFrame(class_rows)


def summarize_per_fold(df: pd.DataFrame, level: str) -> dict:
    out = {
        "level": level,
        "n_folds": int(len(df)),
    }

    for col in ["acc", "macro_f1", "n"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(x) > 0:
                out[f"{col}_mean"] = float(x.mean())
                out[f"{col}_sd"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
                out[f"{col}_min"] = float(x.min())
                out[f"{col}_max"] = float(x.max())
                out[f"{col}_sum"] = float(x.sum()) if col == "n" else np.nan
    return out


def main() -> None:
    args = parse_args()

    per_fold_rows: List[dict] = []
    pooled_rows: List[dict] = []
    class_frames: List[pd.DataFrame] = []

    for lv in [0, 1, 2]:
        level = level_name(lv)
        summary_dir = args.project_root / "outputs" / level_dir_name(lv) / "_summary"

        per_fold_path = summary_dir / "per_fold_metrics.csv"
        cm_path = summary_dir / "confusion_matrix.csv"

        if per_fold_path.exists():
            pf = pd.read_csv(per_fold_path)
            row = summarize_per_fold(pf, level)
            row["per_fold_metrics_path"] = str(per_fold_path)
            per_fold_rows.append(row)
        else:
            per_fold_rows.append(
                {
                    "level": level,
                    "n_folds": 0,
                    "per_fold_metrics_path": str(per_fold_path),
                    "error": "per_fold_metrics.csv not found",
                }
            )

        if cm_path.exists():
            cm_df = pd.read_csv(cm_path, header=None)
            cm = cm_df.to_numpy()
            pooled, cls = metrics_from_confusion(cm, level)
            pooled["confusion_matrix_path"] = str(cm_path)
            pooled_rows.append(pooled)
            class_frames.append(cls)
        else:
            pooled_rows.append(
                {
                    "level": level,
                    "confusion_matrix_path": str(cm_path),
                    "error": "confusion_matrix.csv not found",
                }
            )

    per_fold_summary = pd.DataFrame(per_fold_rows)
    pooled_summary = pd.DataFrame(pooled_rows)
    class_metrics = pd.concat(class_frames, ignore_index=True) if class_frames else pd.DataFrame()

    abstract = per_fold_summary.merge(pooled_summary, on="level", how="outer")

    # Compact abstract-ready columns.
    cols = [
        "level",
        "n_folds",
        "n_sum",
        "acc_mean",
        "macro_f1_mean",
        "n_frames_pooled_confusion",
        "accuracy_pooled_confusion",
        "macro_f1_pooled_confusion",
        "macro_recall_pooled_confusion",
        "recall_min_pooled_confusion",
        "recall_max_pooled_confusion",
        "recall_mean_pooled_confusion",
    ]
    abstract_ready = abstract[[c for c in cols if c in abstract.columns]].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_fold_summary, args.out_dir / "workflow_eval_per_fold_summary.csv", args.overwrite)
    write_csv(pooled_summary, args.out_dir / "workflow_eval_pooled_confusion_summary.csv", args.overwrite)
    write_csv(class_metrics, args.out_dir / "workflow_eval_class_metrics.csv", args.overwrite)
    write_csv(abstract_ready, args.out_dir / "workflow_eval_abstract_ready_summary.csv", args.overwrite)

    print("[DONE] Existing workflow evaluation summarized with headerless confusion matrices.")
    print(f"[INFO] out_dir={args.out_dir}")
    print("\n[ABSTRACT READY SUMMARY]")
    print(abstract_ready.to_string(index=False))

    print("\n[CLASS METRICS: recall range check]")
    if not class_metrics.empty:
        print(
            class_metrics.groupby("level")
            .agg(
                n_classes=("class_index", "count"),
                recall_min=("recall", "min"),
                recall_max=("recall", "max"),
                recall_mean=("recall", "mean"),
                f1_mean=("f1", "mean"),
            )
            .reset_index()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
