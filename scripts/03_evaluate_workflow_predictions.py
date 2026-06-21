#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_workflow_predictions_v3.py

Evaluate annotation_app workflow-classification prediction CSVs.

This version is designed for CSVs with columns:
  image_path, fold, level, pred_id, pred_name, score, gt_id, gt_name

Key fixes from v2:
  - Compares pred_id vs gt_id by default, not pred_name vs gt_name.
    This avoids false 0% accuracy when display names differ.
  - Uses gt_name only for class display and OOB detection.
  - Level-0 is evaluated on all frames.
  - Level-1 and Level-2 are evaluated after excluding frames whose Level-0
    ground truth is OutsideBody, when Level-0 rows can be matched by image_path.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_PRED_DIR = Path("outputs/predictions")
DEFAULT_OUT_DIR = Path("outputs/workflow_prediction_evaluation")

CASE_RE = re.compile(r"(CASE\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate workflow prediction CSVs by pred_id vs gt_id.")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--level0-csv", type=str, default="preds_level0.csv")
    parser.add_argument("--level1-csv", type=str, default="preds_level1.csv")
    parser.add_argument("--level2-csv", type=str, default="preds_level2.csv")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--compare-by",
        choices=["id", "name"],
        default="id",
        help="Default is id. Use name only if pred_name and gt_name are known to be identical label vocabularies.",
    )
    parser.add_argument(
        "--exclude-oob-for-level1-level2",
        action="store_true",
        default=True,
        help="Exclude rows matched to GT Level-0 OutsideBody for Level-1 and Level-2 evaluation.",
    )
    parser.add_argument(
        "--do-not-exclude-oob-for-level1-level2",
        action="store_false",
        dest="exclude_oob_for_level1_level2",
    )
    parser.add_argument("--export-frame-level", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)


def extract_case_id(value: Any) -> str:
    s = safe_str(value)
    m = CASE_RE.search(s)
    return m.group(1).upper() if m else ""


def find_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def infer_case_id_from_row(row: pd.Series) -> str:
    for col in [
        "case_id", "case", "video_id", "video", "video_name",
        "filename", "image_file", "image_path", "path",
    ]:
        if col in row.index:
            case_id = extract_case_id(row[col])
            if case_id:
                return case_id
    return ""


def normalize_level0_label(label: Any) -> str:
    s = safe_str(label).strip()
    low = s.lower()
    if low in ["outsidebody", "outside_body", "outside", "out of body", "oob"] or "outside" in low:
        return "OutsideBody"
    if low in ["insidebody", "inside_body", "inside", "in body"] or "inside" in low:
        return "InsideBody"
    return s


def canonical_id(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    try:
        f = float(x)
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return safe_str(x).strip()


def canonical_name(x: Any) -> str:
    return safe_str(x).strip()


def normalize_prediction_csv(path: Path, expected_level: str, sample_fps: float, compare_by: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(path)

    audit: Dict[str, Any] = {
        "level": expected_level,
        "prediction_csv": str(path),
        "csv_exists": path.exists(),
        "n_rows_raw": int(len(raw)),
        "columns": "|".join(raw.columns.astype(str).tolist()),
        "error": "",
    }

    if raw.empty:
        raise ValueError(f"Prediction CSV is empty: {path}")

    pred_id_col = find_column(raw.columns, ["pred_id", "prediction_id", "pred_class_id", "y_pred_id"])
    gt_id_col = find_column(raw.columns, ["gt_id", "target_id", "true_id", "y_true_id"])
    pred_name_col = find_column(raw.columns, ["pred_name", "pred_label", "predicted_label", "prediction_label", "y_pred", "pred"])
    gt_name_col = find_column(raw.columns, ["gt_name", "gt_label", "ground_truth", "target_name", "true_label", "y_true"])

    if compare_by == "id":
        if pred_id_col is None or gt_id_col is None:
            raise ValueError(f"--compare-by id requires pred_id and gt_id columns. Columns: {raw.columns.tolist()}")
    else:
        if pred_name_col is None or gt_name_col is None:
            raise ValueError(f"--compare-by name requires pred_name and gt_name columns. Columns: {raw.columns.tolist()}")

    case_col = find_column(raw.columns, ["case_id", "case", "video_id", "video", "video_name", "filename", "image_file", "image_path", "path"])
    time_col = find_column(raw.columns, ["sample_time_sec", "time_sec", "timestamp_sec", "sec", "seconds", "time", "timestamp"])
    frame_col = find_column(raw.columns, ["frame_idx_1based", "frame_idx", "frame", "frame_index", "sample_ordinal", "sample_index", "idx"])

    df = pd.DataFrame()

    if case_col is not None:
        df["case_id"] = raw[case_col].map(extract_case_id)
    else:
        df["case_id"] = raw.apply(infer_case_id_from_row, axis=1)

    if "image_path" in raw.columns:
        df["image_path"] = raw["image_path"].astype(str)
    elif "path" in raw.columns:
        df["image_path"] = raw["path"].astype(str)
    else:
        df["image_path"] = ""

    df["frame_key"] = df["image_path"].astype(str)

    if time_col is not None:
        df["sample_time_sec"] = pd.to_numeric(raw[time_col], errors="coerce")
    elif frame_col is not None:
        frame_values = pd.to_numeric(raw[frame_col], errors="coerce")
        df["sample_time_sec"] = frame_values / float(sample_fps)
    else:
        parsed = raw.astype(str).apply(
            lambda r: next((int(m.group(1)) for v in r if (m := re.search(r"frame[_-]?(\d+)", v, re.I))), np.nan),
            axis=1,
        )
        df["sample_time_sec"] = pd.to_numeric(parsed, errors="coerce") / float(sample_fps)
        frame_col = "parsed_from_path"

    if frame_col is not None and frame_col != "parsed_from_path":
        df["frame_index_or_sample_index"] = pd.to_numeric(raw[frame_col], errors="coerce")
    else:
        df["frame_index_or_sample_index"] = np.nan

    empty_key = df["frame_key"].eq("") | df["frame_key"].eq("nan")
    if empty_key.any():
        df.loc[empty_key, "frame_key"] = (
            df.loc[empty_key, "case_id"].astype(str)
            + "__"
            + df.loc[empty_key, "sample_time_sec"].round(3).astype(str)
        )

    df["pred_id"] = raw[pred_id_col].map(canonical_id) if pred_id_col is not None else ""
    df["gt_id"] = raw[gt_id_col].map(canonical_id) if gt_id_col is not None else ""
    df["pred_name"] = raw[pred_name_col].map(canonical_name) if pred_name_col is not None else ""
    df["gt_name"] = raw[gt_name_col].map(canonical_name) if gt_name_col is not None else ""

    if expected_level == "Level-0":
        df["pred_name"] = df["pred_name"].map(normalize_level0_label)
        df["gt_name"] = df["gt_name"].map(normalize_level0_label)

    if compare_by == "id":
        df["pred_key"] = df["pred_id"]
        df["gt_key"] = df["gt_id"]
    else:
        df["pred_key"] = df["pred_name"]
        df["gt_key"] = df["gt_name"]

    df["pred_label"] = df["pred_name"]
    df["gt_label"] = df["gt_name"]

    conf_col = find_column(raw.columns, ["score", "confidence", "conf", "prob", "max_prob"])
    df["pred_confidence"] = pd.to_numeric(raw[conf_col], errors="coerce") if conf_col is not None else np.nan

    df["level"] = expected_level

    before = len(df)
    df = df.loc[df["case_id"] != ""].copy()
    df = df.loc[df["sample_time_sec"].notna()].copy()
    df = df.loc[df["pred_key"] != ""].copy()
    df = df.loc[df["gt_key"] != ""].copy()
    after = len(df)

    audit.update(
        {
            "compare_by": compare_by,
            "pred_id_col": pred_id_col or "",
            "gt_id_col": gt_id_col or "",
            "pred_name_col": pred_name_col or "",
            "gt_name_col": gt_name_col or "",
            "case_col": case_col or "",
            "time_col": time_col or "",
            "frame_col": frame_col or "",
            "confidence_col": conf_col or "",
            "n_rows_normalized": int(after),
            "n_rows_dropped_missing_case_time_or_label": int(before - after),
            "n_cases": int(df["case_id"].nunique()),
            "unique_cases": "|".join(sorted(df["case_id"].unique().tolist())),
        }
    )

    return df, audit


def add_gt_level0_oob(eval_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in eval_dfs.items()}

    if "Level-0" not in out or out["Level-0"].empty:
        for level in out:
            out[level]["gt_level0_label"] = ""
            out[level]["gt_is_oob"] = False
            out[level]["gt_level0_match_found"] = False
        return out

    l0 = out["Level-0"][["frame_key", "case_id", "sample_time_sec", "gt_name"]].copy()
    l0["gt_level0_label"] = l0["gt_name"].map(normalize_level0_label)
    l0 = l0.drop_duplicates(subset=["frame_key"])

    for level, df in out.items():
        if level == "Level-0":
            df["gt_level0_label"] = df["gt_name"].map(normalize_level0_label)
            df["gt_is_oob"] = df["gt_level0_label"] == "OutsideBody"
            df["gt_level0_match_found"] = True
            out[level] = df
            continue

        merged = df.merge(
            l0[["frame_key", "gt_level0_label"]],
            on="frame_key",
            how="left",
            validate="m:1",
        )
        merged["gt_level0_match_found"] = merged["gt_level0_label"].notna()
        merged["gt_level0_label"] = merged["gt_level0_label"].fillna("")
        merged["gt_is_oob"] = merged["gt_level0_label"] == "OutsideBody"
        out[level] = merged

    return out


def display_label_map(df: pd.DataFrame) -> Dict[str, str]:
    rows = []
    for _, r in df.iterrows():
        key = safe_str(r.get("gt_key"))
        name = safe_str(r.get("gt_name"))
        if key and name:
            rows.append((key, name))
        key = safe_str(r.get("pred_key"))
        name = safe_str(r.get("pred_name"))
        if key and name:
            rows.append((key, name))

    if not rows:
        return {}

    tmp = pd.DataFrame(rows, columns=["key", "name"])
    mapping = {}
    for key, sub in tmp.groupby("key"):
        mode = sub["name"].mode()
        mapping[key] = str(mode.iloc[0]) if not mode.empty else str(key)
    return mapping


def class_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    yt = eval_df["gt_key"].astype(str)
    yp = eval_df["pred_key"].astype(str)
    label_map = display_label_map(eval_df)
    keys = sorted(set(yt.unique()).union(set(yp.unique())), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

    rows: List[Dict[str, Any]] = []
    for key in keys:
        tp = int(((yt == key) & (yp == key)).sum())
        fp = int(((yt != key) & (yp == key)).sum())
        fn = int(((yt == key) & (yp != key)).sum())
        support = int((yt == key).sum())
        pred_n = int((yp == key).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else np.nan
        )

        rows.append(
            {
                "class_key": key,
                "class_label": label_map.get(key, key),
                "support": support,
                "predicted_n": pred_n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def summarize_eval_df(df: pd.DataFrame, level: str, evaluation_scope: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    n_prediction_rows = int(len(df))
    eval_df = df.loc[(df["gt_key"].astype(str) != "") & (df["pred_key"].astype(str) != "")].copy()
    n_missing = int(n_prediction_rows - len(eval_df))

    if eval_df.empty:
        return {
            "level": level,
            "evaluation_scope": evaluation_scope,
            "n_prediction_rows": n_prediction_rows,
            "n_evaluated_frames": 0,
            "n_missing_pred_or_gt": n_missing,
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "recall_min": np.nan,
            "recall_max": np.nan,
            "recall_mean": np.nan,
        }, pd.DataFrame()

    correct = eval_df["gt_key"].astype(str) == eval_df["pred_key"].astype(str)
    cls = class_metrics(eval_df)
    supported = cls.loc[cls["support"] > 0].copy()

    summary: Dict[str, Any] = {
        "level": level,
        "evaluation_scope": evaluation_scope,
        "n_prediction_rows": n_prediction_rows,
        "n_evaluated_frames": int(len(eval_df)),
        "n_missing_pred_or_gt": n_missing,
        "accuracy": float(correct.mean()),
        "macro_precision": float(supported["precision"].mean(skipna=True)) if not supported.empty else np.nan,
        "macro_recall": float(supported["recall"].mean(skipna=True)) if not supported.empty else np.nan,
        "macro_f1": float(supported["f1"].mean(skipna=True)) if not supported.empty else np.nan,
        "recall_min": float(supported["recall"].min(skipna=True)) if not supported.empty else np.nan,
        "recall_max": float(supported["recall"].max(skipna=True)) if not supported.empty else np.nan,
        "recall_mean": float(supported["recall"].mean(skipna=True)) if not supported.empty else np.nan,
    }

    if level == "Level-0":
        for label in ["OutsideBody", "InsideBody"]:
            row = cls.loc[cls["class_label"] == label]
            if not row.empty:
                key = label.lower()
                summary[f"{key}_support"] = int(row["support"].iloc[0])
                summary[f"{key}_precision"] = float(row["precision"].iloc[0])
                summary[f"{key}_recall"] = float(row["recall"].iloc[0])
                summary[f"{key}_f1"] = float(row["f1"].iloc[0])

    return summary, cls


def per_case_metrics(df: pd.DataFrame, level: str, evaluation_scope: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_id, sub in df.groupby("case_id", sort=True):
        summary, _ = summarize_eval_df(sub, level, evaluation_scope)
        summary["case_id"] = case_id
        rows.append(summary)
    return pd.DataFrame(rows)


def confusion_counts(eval_df: pd.DataFrame) -> pd.DataFrame:
    label_map = display_label_map(eval_df)

    cm = pd.crosstab(
        eval_df["gt_key"].astype(str),
        eval_df["pred_key"].astype(str),
        rownames=["gt_key"],
        colnames=["pred_key"],
        dropna=False,
    ).reset_index()

    cm.insert(1, "gt_label", cm["gt_key"].map(lambda x: label_map.get(str(x), str(x))))
    return cm


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()

    level_paths = {
        "Level-0": args.pred_dir / args.level0_csv,
        "Level-1": args.pred_dir / args.level1_csv,
        "Level-2": args.pred_dir / args.level2_csv,
    }

    raw_eval_dfs: Dict[str, pd.DataFrame] = {}
    audit_rows: List[Dict[str, Any]] = []

    for level, path in level_paths.items():
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            df, audit = normalize_prediction_csv(path, level, args.sample_fps, args.compare_by)
            raw_eval_dfs[level] = df
            audit_rows.append(audit)
            print(
                f"[OK] read {level}: rows={len(df)} cases={df['case_id'].nunique()} "
                f"compare_by={args.compare_by} cases={audit['unique_cases']}"
            )
        except Exception as e:
            audit_rows.append(
                {
                    "level": level,
                    "prediction_csv": str(path),
                    "csv_exists": path.exists(),
                    "error": str(e),
                }
            )
            print(f"[WARN] {level}: {e}")

    if not raw_eval_dfs:
        raise RuntimeError("No prediction CSVs could be read.")

    eval_dfs = add_gt_level0_oob(raw_eval_dfs)

    overall_rows: List[Dict[str, Any]] = []
    per_case_rows: List[pd.DataFrame] = []
    per_class_rows: List[pd.DataFrame] = []

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for level, df0 in eval_dfs.items():
        df = df0.copy()

        if level in ["Level-1", "Level-2"] and args.exclude_oob_for_level1_level2:
            before = len(df)
            df = df.loc[~df["gt_is_oob"].astype(bool)].copy()
            excluded = before - len(df)
            unmatched = int((~df0["gt_level0_match_found"].astype(bool)).sum()) if "gt_level0_match_found" in df0.columns else np.nan
            scope = f"GT Level-0 OutsideBody excluded when matched; excluded_frames={excluded}; unmatched_level0_frames={unmatched}"
        else:
            scope = "all frames including OutsideBody"

        summary, cls = summarize_eval_df(df, level, scope)
        overall_rows.append(summary)

        if not cls.empty:
            cls.insert(0, "level", level)
            cls.insert(1, "evaluation_scope", scope)
            per_class_rows.append(cls)

        pc = per_case_metrics(df, level, scope)
        if not pc.empty:
            per_case_rows.append(pc)

        eval_non_missing = df.loc[(df["gt_key"].astype(str) != "") & (df["pred_key"].astype(str) != "")].copy()
        if not eval_non_missing.empty:
            cm = confusion_counts(eval_non_missing)
            write_csv(cm, args.out_dir / f"confusion_{level.lower().replace('-', '')}.csv", args.overwrite)

        if args.export_frame_level:
            write_csv(df, args.out_dir / f"frame_level_eval_{level.lower().replace('-', '')}.csv", args.overwrite)

        print(
            f"[OK] eval {level}: evaluated={summary.get('n_evaluated_frames')} "
            f"accuracy={summary.get('accuracy')} macro_f1={summary.get('macro_f1')}"
        )

    overall_df = pd.DataFrame(overall_rows)
    per_case_df = pd.concat(per_case_rows, ignore_index=True) if per_case_rows else pd.DataFrame()
    per_class_df = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    audit_df = pd.DataFrame(audit_rows)

    write_csv(overall_df, args.out_dir / "overall_metrics.csv", args.overwrite)
    write_csv(per_case_df, args.out_dir / "per_case_metrics.csv", args.overwrite)
    write_csv(per_class_df, args.out_dir / "per_class_metrics.csv", args.overwrite)
    write_csv(audit_df, args.out_dir / "input_file_audit.csv", args.overwrite)

    print("[DONE] Workflow prediction evaluation exported")
    print(f"[INFO] out_dir={args.out_dir}")
    for name in [
        "overall_metrics.csv",
        "per_case_metrics.csv",
        "per_class_metrics.csv",
        "confusion_level0.csv",
        "confusion_level1.csv",
        "confusion_level2.csv",
        "input_file_audit.csv",
    ]:
        path = args.out_dir / name
        if path.exists():
            print(f"  - {path}")


if __name__ == "__main__":
    main()
