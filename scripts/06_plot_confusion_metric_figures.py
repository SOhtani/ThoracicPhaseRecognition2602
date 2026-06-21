#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
plot_workflow_confusion_metric_figures_fixed.py

Create one publication-style metric figure per workflow level.

Input files are existing evaluation results, not newly computed predictions:
  Level-0:
    outputs/level0/_summary/confusion_matrix.csv
    outputs/level0/_summary/per_fold_metrics.csv
  Level-1:
    outputs/level1/_summary/confusion_matrix.csv
    outputs/level1/_summary/per_fold_metrics.csv
  Level-2, default old 12-step rule:
    outputs/level2_oldrule_backup_20251224_050408/_summary/confusion_matrix.csv
    outputs/level2_oldrule_backup_20251224_050408/_summary/per_fold_metrics.csv

The confusion_matrix.csv files in this project are headerless matrices.
They are read with header=None.

Outputs:
  - level0_confusion_metrics.png / svg / pdf
  - level1_confusion_metrics.png / svg / pdf
  - level2_confusion_metrics.png / svg / pdf
  - figure_metric_summary.csv

Requires:
  python -m pip install matplotlib pandas numpy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_OUT_DIR = Path("outputs/publication_metric_figures")
DEFAULT_LEVEL2_OLD_DIR = DEFAULT_PROJECT_ROOT / "outputs" / "level2_oldrule_backup"
DEFAULT_LEVEL2_CURRENT_DIR = DEFAULT_PROJECT_ROOT / "outputs" / "level2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot publication-style row-normalized confusion matrices and summary metrics."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--level2-source",
        choices=["oldrule", "current"],
        default="oldrule",
        help="Use old 12-step Level-2 results or current Level-2 results.",
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        default=["png", "svg", "pdf"],
        choices=["png", "svg", "pdf"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--max-label-length", type=int, default=24)
    parser.add_argument(
        "--show-fold-mean-row",
        action="store_true",
        help="Add a second table row for fold-mean accuracy and macro F1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    """Write CSV with overwrite protection."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_confusion_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Confusion matrix not found: {path}")
    df = pd.read_csv(path, header=None)
    cm = df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[1] == 0:
        raise ValueError(f"Invalid confusion matrix shape at {path}: {cm.shape}")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"Confusion matrix must be square. Got {cm.shape} at {path}. "
            "Check whether the file was read correctly."
        )
    return cm


def read_per_fold_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"per_fold_metrics not found: {path}")
    return pd.read_csv(path)


def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def metrics_from_confusion(cm: np.ndarray) -> Tuple[Dict[str, float], pd.DataFrame]:
    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    diag = np.diag(cm)
    support = cm.sum(axis=1)
    predicted_n = cm.sum(axis=0)

    precision = np.divide(
        diag,
        predicted_n,
        out=np.full_like(diag, np.nan, dtype=float),
        where=predicted_n != 0,
    )
    recall = np.divide(
        diag,
        support,
        out=np.full_like(diag, np.nan, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full_like(diag, np.nan, dtype=float),
        where=np.isfinite(precision + recall) & ((precision + recall) != 0),
    )

    supported = support > 0

    metrics = {
        "n_frames": int(total),
        "accuracy": safe_div(float(diag.sum()), total),
        "macro_precision": float(np.nanmean(precision[supported])),
        "macro_recall": float(np.nanmean(recall[supported])),
        "macro_f1": float(np.nanmean(f1[supported])),
        "recall_min": float(np.nanmin(recall[supported])),
        "recall_max": float(np.nanmax(recall[supported])),
        "recall_mean": float(np.nanmean(recall[supported])),
    }

    class_rows: List[Dict[str, float]] = []
    for i in range(cm.shape[0]):
        class_rows.append(
            {
                "class_index": i,
                "support": int(support[i]),
                "predicted_n": int(predicted_n[i]),
                "precision": float(precision[i]) if np.isfinite(precision[i]) else np.nan,
                "recall": float(recall[i]) if np.isfinite(recall[i]) else np.nan,
                "f1": float(f1[i]) if np.isfinite(f1[i]) else np.nan,
            }
        )

    return metrics, pd.DataFrame(class_rows)


def fold_mean_metrics(per_fold: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "acc" in per_fold.columns:
        out["acc_mean"] = float(pd.to_numeric(per_fold["acc"], errors="coerce").mean())
        out["acc_sd"] = float(pd.to_numeric(per_fold["acc"], errors="coerce").std(ddof=1))
    if "macro_f1" in per_fold.columns:
        out["macro_f1_mean"] = float(pd.to_numeric(per_fold["macro_f1"], errors="coerce").mean())
        out["macro_f1_sd"] = float(pd.to_numeric(per_fold["macro_f1"], errors="coerce").std(ddof=1))
    if "n" in per_fold.columns:
        out["n_sum"] = int(pd.to_numeric(per_fold["n"], errors="coerce").sum())
    out["n_folds"] = int(len(per_fold))
    return out


def load_label_map(path: Path) -> Dict[int, str]:
    """
    Supports several common label_map.json structures:
      {"0": "label"}
      {"label": 0}
      {"idx_to_label": {"0": "label"}}
      {"label_to_idx": {"label": 0}}
      [{"id": 0, "name": "label"}, ...]
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    mapping: Dict[int, str] = {}

    def add_pair(k, v) -> None:
        try:
            ik = int(k)
            mapping[ik] = str(v)
            return
        except Exception:
            pass
        try:
            iv = int(v)
            mapping[iv] = str(k)
            return
        except Exception:
            pass

    if isinstance(data, dict):
        for key in ["idx_to_label", "id_to_label", "index_to_label", "classes"]:
            obj = data.get(key)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    add_pair(k, v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if isinstance(v, str):
                        mapping[i] = v

        for key in ["label_to_idx", "label2id", "class_to_idx"]:
            obj = data.get(key)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    add_pair(k, v)

        if not mapping:
            for k, v in data.items():
                if isinstance(v, (str, int, float)):
                    add_pair(k, v)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                mapping[i] = item
            elif isinstance(item, dict):
                idx = item.get("id", item.get("index", item.get("class_index", i)))
                name = item.get("name", item.get("label", item.get("class_label", str(idx))))
                try:
                    mapping[int(idx)] = str(name)
                except Exception:
                    pass

    return mapping


def truncate_label(s: str, max_len: int) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def default_label_map_for_level(level: str, n_classes: int, label_map_path: Optional[Path], max_len: int) -> List[str]:
    mapping = load_label_map(label_map_path) if label_map_path is not None else {}

    labels = []
    for i in range(n_classes):
        labels.append(truncate_label(mapping.get(i, str(i)), max_len=max_len))
    return labels


def figure_paths(out_dir: Path, level_stub: str, formats: List[str]) -> List[Path]:
    return [out_dir / f"{level_stub}_confusion_metrics.{fmt}" for fmt in formats]


def add_metric_table(ax, rows: List[List[str]], col_labels: List[str]) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.2)


def plot_level_figure(
    level: str,
    cm: np.ndarray,
    per_fold: pd.DataFrame,
    labels: List[str],
    out_paths: List[Path],
    dpi: int,
    cmap: str,
    show_fold_mean_row: bool,
) -> Dict[str, float]:
    norm = row_normalize(cm)
    metrics, _class_metrics = metrics_from_confusion(cm)
    fold_metrics = fold_mean_metrics(per_fold)

    n_classes = cm.shape[0]
    fig_width = max(7.0, 0.65 * n_classes + 3.0)
    fig_height = max(7.0, 0.55 * n_classes + 3.2)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[5.2, 1.25],
        width_ratios=[20, 1],
        hspace=0.30,
        wspace=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    tax = fig.add_subplot(gs[1, :])

    im = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)

    ax.set_title(f"{level} pooled cross-validation confusion matrix", fontsize=13, pad=14)
    ax.set_xlabel("Prediction label", fontsize=11)
    ax.set_ylabel("Ground truth label", fontsize=11)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=45 if n_classes > 6 else 0, ha="right" if n_classes > 6 else "center")
    ax.set_yticklabels(labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_classes):
        for j in range(n_classes):
            value = norm[i, j]
            text_color = "white" if value < 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    for i in range(n_classes):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.1)
        ax.add_patch(rect)

    rows = [
        [
            "Pooled CV",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['macro_precision']:.3f}",
            f"{metrics['macro_recall']:.3f}",
            f"{metrics['macro_f1']:.3f}",
            f"{metrics['n_frames']:,}",
        ]
    ]

    if show_fold_mean_row:
        n_sum = fold_metrics.get("n_sum", np.nan)
        n_sum_text = f"{n_sum:,.0f}" if np.isfinite(n_sum) else ""
        rows.append(
            [
                "Fold mean",
                f"{fold_metrics.get('acc_mean', np.nan):.3f}",
                "",
                "",
                f"{fold_metrics.get('macro_f1_mean', np.nan):.3f}",
                n_sum_text,
            ]
        )

    add_metric_table(
        tax,
        rows=rows,
        col_labels=["Dataset", "Accuracy", "Precision", "Recall", "F1 score", "Frames"],
    )

    fig.subplots_adjust(left=0.13, right=0.90, top=0.88, bottom=0.10)

    for out_path in out_paths:
        if out_path.suffix.lower() == ".png":
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")

    plt.close(fig)

    summary = {
        "level": level,
        **metrics,
        **{f"fold_{k}": v for k, v in fold_metrics.items()},
    }

    return summary


def get_level_config(project_root: Path, level2_source: str) -> Dict[str, Dict[str, Path]]:
    outputs = project_root / "outputs"

    level2_dir = (
        project_root / "outputs" / "level2_oldrule_backup_20251224_050408"
        if level2_source == "oldrule"
        else project_root / "outputs" / "level2"
    )

    return {
        "Level-0": {
            "summary_dir": outputs / "level0" / "_summary",
            "label_map": outputs / "level0" / "cv0" / "_normalized" / "label_map.json",
        },
        "Level-1": {
            "summary_dir": outputs / "level1" / "_summary",
            "label_map": outputs / "level1" / "cv0" / "_normalized" / "label_map.json",
        },
        "Level-2": {
            "summary_dir": level2_dir / "_summary",
            "label_map": level2_dir / "cv0" / "_normalized" / "label_map.json",
        },
    }


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = get_level_config(args.project_root, args.level2_source)

    summary_rows: List[Dict[str, float]] = []
    written: List[Path] = []

    for level, cfg in configs.items():
        summary_dir = cfg["summary_dir"]
        cm_path = summary_dir / "confusion_matrix.csv"
        pf_path = summary_dir / "per_fold_metrics.csv"

        cm = read_confusion_matrix(cm_path)
        per_fold = read_per_fold_metrics(pf_path)
        labels = default_label_map_for_level(
            level=level,
            n_classes=cm.shape[0],
            label_map_path=cfg.get("label_map"),
            max_len=args.max_label_length,
        )

        level_stub = level.lower().replace("-", "")
        out_paths = figure_paths(args.out_dir, level_stub, args.save_formats)

        if not args.overwrite:
            existing = [p for p in out_paths if p.exists()]
            if existing:
                raise FileExistsError(f"Output exists. Use --overwrite: {existing[0]}")

        summary = plot_level_figure(
            level=level,
            cm=cm,
            per_fold=per_fold,
            labels=labels,
            out_paths=out_paths,
            dpi=args.dpi,
            cmap=args.cmap,
            show_fold_mean_row=args.show_fold_mean_row,
        )
        summary["confusion_matrix_path"] = str(cm_path)
        summary["per_fold_metrics_path"] = str(pf_path)
        summary["label_map_path"] = str(cfg.get("label_map"))

        summary_rows.append(summary)
        written.extend(out_paths)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.out_dir / "figure_metric_summary.csv"
    write_csv(summary_df, summary_csv, args.overwrite)
    written.append(summary_csv)

    print("[DONE] Publication-style workflow metric figures created.")
    print(f"[INFO] level2_source={args.level2_source}")
    print(f"[INFO] out_dir={args.out_dir}")
    print("[INFO] wrote:")
    for p in written:
        print(f"  - {p}")
    print("[INFO] summary:")
    compact_cols = [
        "level",
        "n_frames",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "recall_min",
        "recall_max",
        "recall_mean",
    ]
    print(summary_df[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
