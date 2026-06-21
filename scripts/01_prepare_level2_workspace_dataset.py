#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_new_l2_newfold_from_v2_json.py

Prepare new active Level-2 field/workspace data for the existing workflow model
pipeline using only CASE###_annotation_v2.json files.

This version is for the current project layout:
  /path/to/project/data/raw/videos/CASE001_annotation_v2.json

Key points:
  - It only uses *_annotation_v2.json by default.
  - It does not use migrated_annotations.
  - It does not use old CASE###_annotation.json unless --annotation-version-pattern is changed.
  - It extracts new L2 segments robustly from multiple possible JSON structures:
      active_annotation.segments
      segments
      layers[].segments
      layers[].items
      any nested dict with label + start/end fields
  - It filters to allowed new L2 field/workspace labels by default.
  - It writes full audits before failing.

Output layout follows the existing workflow model convention:
  outputs/new_level2_newfold/cv0/_normalized/train_cv0.txt
  outputs/new_level2_newfold/cv0/_normalized/test_cv0.txt
  outputs/new_level2_newfold/cv0/_normalized/label_map.json

Each list line:
  CASE###/frame_000001.png <class_index>
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_NEW_L2_LABELS = [
    "inferior_hilum",
    "posterior_hilum",
    "superior_hilum",
    "anterior_hilum",
    "fissure",
    "subcarinal",
    "superior_mediastinum",
    "uncertain",
    "FireFly",
    "InterSegmental",
]

OLD_L2_HINT_LABELS = {
    "Lymphadenectomy",
    "Fissure",
    "PA_handle",
    "Bronchus_handle",
    "mediastinal_pleural_incision",
    "PV_handle",
    "Pulmonary_ligament_dissecion",
    "Pulmonary_ligament_disection",
    "Pulmonary_ligament_dissection",
    "Specimen_retrieval",
    "Adhesion",
    "upper_mediastinal_lymphnode",
    "#7_dissection",
    "Intersegment",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare new Level-2 dataset from CASE###_annotation_v2.json files."
    )
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--frame-csv-dir", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-level-dir", type=Path, default=None)

    p.add_argument("--case-start", type=str, default="CASE001")
    p.add_argument("--case-end", type=str, default="CASE020")
    p.add_argument("--cases", nargs="*", default=None)

    p.add_argument(
        "--annotation-version-pattern",
        type=str,
        default="{case_id}_annotation_v2.json",
        help="Filename pattern relative to annotation-root recursive search. Default: {case_id}_annotation_v2.json",
    )

    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--split-strategy", choices=["balanced_frames", "round_robin"], default="balanced_frames")

    p.add_argument("--allowed-labels", nargs="*", default=None)
    p.add_argument("--include-uncertain", action="store_true")
    p.add_argument("--include-special", action="store_true")
    p.add_argument("--allow-unknown-labels", action="store_true")
    p.add_argument("--min-segment-sec", type=float, default=0.0)
    p.add_argument("--on-overlap", choices=["skip", "first"], default="skip")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_case_id(x: Any) -> str:
    m = re.search(r"(CASE\d{3})", str(x), flags=re.IGNORECASE)
    return m.group(1).upper() if m else str(x).upper()


def case_range(start: str, end: str) -> List[str]:
    s = int(re.search(r"(\d{3})", start).group(1))
    e = int(re.search(r"(\d{3})", end).group(1))
    return [f"CASE{i:03d}" for i in range(s, e + 1)]


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def segment_duration(row: Dict[str, Any]) -> float:
    start = to_float_or_none(row.get("start_sec"))
    end = to_float_or_none(row.get("end_sec"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def get_allowed_labels(args: argparse.Namespace) -> List[str]:
    allowed = list(args.allowed_labels) if args.allowed_labels else list(DEFAULT_NEW_L2_LABELS)
    if not args.include_uncertain and "uncertain" in allowed:
        allowed.remove("uncertain")
    if not args.include_special:
        allowed = [x for x in allowed if x not in {"FireFly", "InterSegmental"}]
    return allowed


def pick_value(d: Dict[str, Any], candidates: List[str]) -> Any:
    for c in candidates:
        if c in d:
            return d.get(c)
    return None


def normalize_segment_dict(obj: Dict[str, Any], json_path: str) -> Optional[Dict[str, Any]]:
    """Return standardized segment dict if object looks like a time segment with a label."""
    label = pick_value(obj, ["label", "name", "class_label", "class_name", "phase", "value"])
    if label is None:
        return None

    level = pick_value(obj, ["level", "level_name", "layer", "layer_name", "type"])
    start = pick_value(obj, ["start_sec", "start", "start_time", "t_start", "begin_sec", "begin"])
    end = pick_value(obj, ["end_sec", "end", "end_time", "t_end", "stop_sec", "stop"])

    start_f = to_float_or_none(start)
    end_f = to_float_or_none(end)

    if start_f is None or end_f is None:
        return None
    if end_f <= start_f:
        return None

    return {
        "level": "" if level is None else str(level),
        "label": str(label),
        "start_sec": start_f,
        "end_sec": end_f,
        "duration_sec": max(0.0, end_f - start_f),
        "json_path": json_path,
    }


def walk_segments(obj: Any, path: List[str], out: List[Dict[str, Any]]) -> None:
    path_str = ".".join(path)

    if isinstance(obj, dict):
        seg = normalize_segment_dict(obj, path_str)
        if seg is not None:
            out.append(seg)

        for k, v in obj.items():
            walk_segments(v, path + [str(k)], out)

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_segments(v, path + [str(i)], out)


def extract_candidate_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract candidate time segments from known and unknown JSON layouts.
    Duplicates may arise from walking; they are removed later.
    """
    rows: List[Dict[str, Any]] = []

    # Known v2 style.
    if isinstance(data.get("active_annotation"), dict):
        for i, seg in enumerate(data.get("active_annotation", {}).get("segments", []) or []):
            if isinstance(seg, dict):
                norm = normalize_segment_dict(seg, ["active_annotation", "segments", str(i)].__str__())
                if norm is not None:
                    norm["json_path"] = f"active_annotation.segments.{i}"
                    rows.append(norm)

    # Legacy/simple style.
    for i, seg in enumerate(data.get("segments", []) or []):
        if isinstance(seg, dict):
            norm = normalize_segment_dict(seg, f"segments.{i}")
            if norm is not None:
                rows.append(norm)

    # Layered style.
    for li, layer in enumerate(data.get("layers", []) or []):
        if not isinstance(layer, dict):
            continue

        layer_name = str(layer.get("name", layer.get("id", "")))
        for field in ["segments", "items", "annotations", "data"]:
            for si, seg in enumerate(layer.get(field, []) or []):
                if not isinstance(seg, dict):
                    continue
                norm = normalize_segment_dict(seg, f"layers.{li}.{field}.{si}")
                if norm is not None:
                    if not norm.get("level"):
                        norm["level"] = layer_name
                    rows.append(norm)

    # Fallback recursive search.
    walk_segments(data, [], rows)

    # Deduplicate exact same label/time/path-insensitive rows.
    dedup = {}
    for r in rows:
        key = (
            str(r.get("level", "")),
            str(r.get("label", "")),
            round(float(r.get("start_sec", 0.0)), 6),
            round(float(r.get("end_sec", 0.0)), 6),
        )
        if key not in dedup:
            dedup[key] = r
        else:
            # Prefer explicit known path over recursive root path.
            old = dedup[key]
            if len(str(r.get("json_path", ""))) < len(str(old.get("json_path", ""))):
                dedup[key] = r

    return list(dedup.values())


def should_treat_as_new_l2(seg: Dict[str, Any], allowed: set, allow_unknown: bool) -> bool:
    label = str(seg.get("label", ""))

    if label in allowed:
        return True

    if allow_unknown and label not in OLD_L2_HINT_LABELS:
        return True

    return False


def find_case_v2_json(annotation_root: Path, case_id: str, pattern: str) -> Optional[Path]:
    filename = pattern.format(case_id=case_id)
    direct = annotation_root / filename
    if direct.exists():
        return direct

    matches = list(annotation_root.rglob(filename))
    if not matches:
        return None

    # Prefer shortest path, then newest.
    matches = sorted(matches, key=lambda p: (len(str(p)), -p.stat().st_mtime))
    return matches[0]


def load_l2_segments(
    annotation_root: Path,
    cases: List[str],
    pattern: str,
    min_segment_sec: float,
    allowed_labels: List[str],
    allow_unknown_labels: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows: List[Dict[str, Any]] = []
    included_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    allowed = set(allowed_labels)

    for case_id in cases:
        p = find_case_v2_json(annotation_root, case_id, pattern)

        if p is None:
            selected_rows.append({
                "case_id": case_id,
                "found": False,
                "selected_annotation_json": "",
                "n_candidate_segments": 0,
                "new_l2_segments_allowed": 0,
                "new_l2_segments_excluded": 0,
                "new_l2_labels_allowed": "",
                "all_labels_seen": "",
            })
            continue

        data = safe_load_json(p)
        if data is None:
            selected_rows.append({
                "case_id": case_id,
                "found": False,
                "selected_annotation_json": str(p),
                "n_candidate_segments": 0,
                "new_l2_segments_allowed": 0,
                "new_l2_segments_excluded": 0,
                "new_l2_labels_allowed": "",
                "all_labels_seen": "",
            })
            continue

        candidates = extract_candidate_segments(data)
        candidates = [s for s in candidates if segment_duration(s) >= min_segment_sec]

        n_allowed = 0
        n_excluded = 0
        allowed_seen = []
        all_seen = []

        for seg in candidates:
            label = str(seg.get("label", ""))
            all_seen.append(label)

            row = {
                "case_id": case_id,
                "label": label,
                "level": str(seg.get("level", "")),
                "start_sec": float(seg.get("start_sec")),
                "end_sec": float(seg.get("end_sec")),
                "duration_sec": float(seg.get("duration_sec")),
                "json_path": str(seg.get("json_path", "")),
                "source_json": str(p),
            }
            all_rows.append(row)

            if should_treat_as_new_l2(seg, allowed, allow_unknown_labels):
                inc = dict(row)
                inc["new_l2_label"] = inc.pop("label")
                included_rows.append(inc)
                n_allowed += 1
                allowed_seen.append(label)
            else:
                exc = dict(row)
                exc["exclude_reason"] = "label_not_allowed_new_l2_or_old_l2_hint"
                excluded_rows.append(exc)
                n_excluded += 1

        selected_rows.append({
            "case_id": case_id,
            "found": True,
            "selected_annotation_json": str(p),
            "n_candidate_segments": len(candidates),
            "new_l2_segments_allowed": n_allowed,
            "new_l2_segments_excluded": n_excluded,
            "new_l2_labels_allowed": ";".join(sorted(set(allowed_seen))),
            "all_labels_seen": ";".join(sorted(set(all_seen))),
        })

    return (
        pd.DataFrame(selected_rows),
        pd.DataFrame(included_rows),
        pd.DataFrame(excluded_rows),
        pd.DataFrame(all_rows),
    )


def image_rel_from_row(row: pd.Series) -> str:
    case_id = str(row["case_id"])
    if "image_path" in row.index and pd.notna(row["image_path"]):
        s = str(row["image_path"]).replace("\\", "/")
        m = re.search(r"(CASE\d{3}/[^/]+)$", s, flags=re.IGNORECASE)
        if m:
            rel = m.group(1)
            return normalize_case_id(rel) + "/" + rel.split("/")[-1]

    if "image_file" in row.index and pd.notna(row["image_file"]) and str(row["image_file"]).strip():
        return f"{case_id}/{Path(str(row['image_file'])).name}"

    idx = int(float(row["frame_idx_1based"]))
    return f"{case_id}/frame_{idx:06d}.png"


def load_frames(frame_csv_dir: Path, cases: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    missing_rows = []

    for case_id in cases:
        p = frame_csv_dir / f"{case_id}_frame_scores.csv"

        if not p.exists():
            missing_rows.append({"case_id": case_id, "missing_frame_csv": str(p)})
            continue

        df = pd.read_csv(p)

        if "case_id" not in df.columns:
            df["case_id"] = case_id
        else:
            df["case_id"] = df["case_id"].map(normalize_case_id)

        if "sample_time_sec" not in df.columns or "frame_idx_1based" not in df.columns:
            raise ValueError(f"{p} must contain sample_time_sec and frame_idx_1based")

        keep = ["case_id", "sample_time_sec", "frame_idx_1based", "image_path", "image_file"]
        keep = [c for c in keep if c in df.columns]
        rows.append(df[keep].copy())

    if not rows:
        return pd.DataFrame(), pd.DataFrame(missing_rows)

    out = pd.concat(rows, ignore_index=True)
    out["sample_time_sec"] = pd.to_numeric(out["sample_time_sec"], errors="coerce")
    out["frame_idx_1based"] = pd.to_numeric(out["frame_idx_1based"], errors="coerce")
    out = out.dropna(subset=["case_id", "sample_time_sec", "frame_idx_1based"]).copy()
    out = out.sort_values(["case_id", "sample_time_sec", "frame_idx_1based"], kind="mergesort").reset_index(drop=True)
    out["image_rel_path"] = out.apply(image_rel_from_row, axis=1)

    return out, pd.DataFrame(missing_rows)


def attach_l2_labels(frames: pd.DataFrame, segments: pd.DataFrame, on_overlap: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = frames.copy()
    out["new_l2_label"] = ""
    out["n_new_l2_labels_at_frame"] = 0

    if frames.empty or segments.empty:
        return out, pd.DataFrame()

    for s in segments.itertuples(index=False):
        mask = (
            out["case_id"].eq(getattr(s, "case_id"))
            & out["sample_time_sec"].ge(float(getattr(s, "start_sec")))
            & out["sample_time_sec"].lt(float(getattr(s, "end_sec")))
        )

        if not mask.any():
            continue

        lab = str(getattr(s, "new_l2_label"))
        existing = out.loc[mask, "new_l2_label"].astype(str)
        empty = existing.eq("") | existing.eq("nan")
        out.loc[mask & empty, "new_l2_label"] = lab
        out.loc[mask & ~empty, "new_l2_label"] = existing[~empty] + "|" + lab
        out.loc[mask, "n_new_l2_labels_at_frame"] = out.loc[mask, "new_l2_label"].astype(str).str.split("|").str.len()

    overlap = out[out["new_l2_label"].astype(str).str.contains(r"\|", regex=True)].copy()

    if on_overlap == "skip":
        out = out[~out["new_l2_label"].astype(str).str.contains(r"\|", regex=True)].copy()
    elif on_overlap == "first":
        out["new_l2_label"] = out["new_l2_label"].astype(str).str.split("|").str[0]

    return out, overlap


def make_label_order(allowed_labels: List[str], labels_seen: List[str], allow_unknown_labels: bool) -> List[str]:
    order = [x for x in allowed_labels if x in set(labels_seen)]
    if allow_unknown_labels:
        for lab in sorted(set(labels_seen)):
            if lab and lab not in order:
                order.append(lab)
    return order


def assign_new_case_folds(labeled: pd.DataFrame, n_folds: int, random_seed: int, strategy: str) -> pd.DataFrame:
    case_summary = (
        labeled.groupby("case_id")
        .agg(
            n_labeled_frames=("image_rel_path", "size"),
            n_fields=("new_l2_label", "nunique"),
        )
        .reset_index()
    )

    n_cases = int(len(case_summary))
    if n_cases <= 0:
        raise RuntimeError("No cases with labeled frames.")

    n_folds_eff = min(int(n_folds), n_cases)
    if n_folds_eff < int(n_folds):
        print(f"[WARN] n_cases={n_cases} < requested n_folds={n_folds}; using n_folds={n_folds_eff}")

    rng = random.Random(int(random_seed))

    if strategy == "round_robin":
        case_ids = case_summary["case_id"].tolist()
        rng.shuffle(case_ids)
        fold_map = {case_id: i % n_folds_eff for i, case_id in enumerate(case_ids)}
    else:
        rows = case_summary.to_dict("records")
        rng.shuffle(rows)
        rows = sorted(rows, key=lambda r: int(r["n_labeled_frames"]), reverse=True)

        fold_loads = {i: 0 for i in range(n_folds_eff)}
        fold_case_counts = {i: 0 for i in range(n_folds_eff)}
        fold_map = {}

        for row in rows:
            fold = min(fold_loads, key=lambda k: (fold_loads[k], fold_case_counts[k], k))
            case_id = row["case_id"]
            fold_map[case_id] = int(fold)
            fold_loads[fold] += int(row["n_labeled_frames"])
            fold_case_counts[fold] += 1

    case_summary["fold"] = case_summary["case_id"].map(fold_map).astype(int)
    return case_summary.sort_values(["fold", "case_id"])


def write_dataset_lists(
    labeled: pd.DataFrame,
    label_order: List[str],
    case_folds: pd.DataFrame,
    out_level_dir: Path,
    overwrite: bool,
) -> None:
    if out_level_dir.exists() and any(out_level_dir.iterdir()):
        if overwrite:
            shutil.rmtree(out_level_dir)
        else:
            raise FileExistsError(f"Output level dir exists. Use --overwrite: {out_level_dir}")

    out_level_dir.mkdir(parents=True, exist_ok=True)

    label_to_idx = {lab: i for i, lab in enumerate(label_order)}
    fold_map = dict(zip(case_folds["case_id"], case_folds["fold"]))

    df = labeled[labeled["new_l2_label"].isin(label_to_idx)].copy()
    df["class_index"] = df["new_l2_label"].map(label_to_idx).astype(int)
    df["fold"] = df["case_id"].map(fold_map).astype(int)

    df.to_csv(out_level_dir / "new_l2_frame_labels_for_model.csv", index=False, encoding="utf-8-sig")

    max_fold = int(case_folds["fold"].max()) if not case_folds.empty else -1

    for fold in range(max_fold + 1):
        cv_dir = out_level_dir / f"cv{fold}" / "_normalized"
        cv_dir.mkdir(parents=True, exist_ok=True)

        train = df[df["fold"] != fold].copy()
        test = df[df["fold"] == fold].copy()

        for name, sub in [("train", train), ("test", test)]:
            path = cv_dir / f"{name}_cv{fold}.txt"
            with path.open("w", encoding="utf-8", newline="\n") as f:
                for r in sub.itertuples(index=False):
                    f.write(f"{getattr(r, 'image_rel_path')} {int(getattr(r, 'class_index'))}\n")

        label_map = {
            "global_labels": list(range(len(label_order))),
            "label_map": {str(i): i for i in range(len(label_order))},
            "idx_to_label": {str(i): lab for i, lab in enumerate(label_order)},
            "label_to_idx": {lab: i for i, lab in enumerate(label_order)},
            "num_classes": len(label_order),
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "fold_strategy": "new_case_level_split_from_v2_json",
        }
        (cv_dir / "label_map.json").write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    case_folds.to_csv(out_level_dir / "new_l2_case_folds.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "level": "New-Level-2",
            "class_index": list(range(len(label_order))),
            "class_label": label_order,
        }
    ).to_csv(out_level_dir / "new_l2_label_names.csv", index=False, encoding="utf-8-sig")

    dataset_class_summary = (
        df.groupby(["new_l2_label", "class_index"])
        .agg(n_frames=("image_rel_path", "size"), n_cases=("case_id", "nunique"))
        .reset_index()
        .sort_values("class_index")
    )
    dataset_class_summary.to_csv(out_level_dir / "new_l2_dataset_class_summary.csv", index=False, encoding="utf-8-sig")

    fold_class_summary = (
        df.groupby(["fold", "new_l2_label", "class_index"])
        .agg(n_frames=("image_rel_path", "size"), n_cases=("case_id", "nunique"))
        .reset_index()
        .sort_values(["fold", "class_index"])
    )
    fold_class_summary.to_csv(out_level_dir / "new_l2_fold_class_summary.csv", index=False, encoding="utf-8-sig")

    fold_summary = (
        df.groupby("fold")
        .agg(
            n_frames=("image_rel_path", "size"),
            n_cases=("case_id", "nunique"),
            n_classes=("class_index", "nunique"),
        )
        .reset_index()
        .sort_values("fold")
    )
    fold_summary.to_csv(out_level_dir / "new_l2_fold_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()

    cases = [normalize_case_id(x) for x in args.cases] if args.cases else case_range(args.case_start, args.case_end)
    out_level_dir = args.out_level_dir or (args.project_root / "outputs" / "new_level2_newfold")
    out_level_dir.parent.mkdir(parents=True, exist_ok=True)

    allowed_labels = get_allowed_labels(args)

    print("[INFO] Preparing new Level-2 data from v2 JSON files")
    print(f"[INFO] annotation_root={args.annotation_root}")
    print(f"[INFO] annotation_version_pattern={args.annotation_version_pattern}")
    print(f"[INFO] requested cases={cases[0]}..{cases[-1]} n={len(cases)}")
    print(f"[INFO] split_strategy={args.split_strategy} n_folds={args.n_folds} random_seed={args.random_seed}")
    print(f"[INFO] allowed_labels={allowed_labels}")
    print(f"[INFO] out_level_dir={out_level_dir}")

    selected, segments, excluded_segments, all_segments = load_l2_segments(
        annotation_root=args.annotation_root,
        cases=cases,
        pattern=args.annotation_version_pattern,
        min_segment_sec=float(args.min_segment_sec),
        allowed_labels=allowed_labels,
        allow_unknown_labels=bool(args.allow_unknown_labels),
    )

    selected_path = out_level_dir.parent / "new_level2_newfold_selected_v2_json_files.csv"
    all_path = out_level_dir.parent / "new_level2_newfold_all_segments_from_v2_json_audit.csv"
    included_path = out_level_dir.parent / "new_level2_newfold_included_segments_for_model.csv"
    excluded_path = out_level_dir.parent / "new_level2_newfold_excluded_segments_audit.csv"

    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    all_segments.to_csv(all_path, index=False, encoding="utf-8-sig")
    segments.to_csv(included_path, index=False, encoding="utf-8-sig")
    excluded_segments.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    frames, missing_frame_csv = load_frames(args.frame_csv_dir, cases)
    missing_frame_csv.to_csv(out_level_dir.parent / "new_level2_newfold_missing_frame_csv_audit.csv", index=False, encoding="utf-8-sig")

    labeled, overlap = attach_l2_labels(frames, segments, args.on_overlap)
    overlap.to_csv(out_level_dir.parent / "new_level2_newfold_overlap_frames_audit.csv", index=False, encoding="utf-8-sig")

    labeled = labeled[labeled["new_l2_label"].astype(str).ne("")].copy()

    if labeled.empty:
        print("[ERROR] No usable labeled frames found.")
        print(f"[INFO] selected audit: {selected_path}")
        print(f"[INFO] all segment audit: {all_path}")
        print(f"[INFO] included segments: {included_path}")
        print(f"[INFO] excluded segments: {excluded_path}")
        print("[INFO] Selected v2 JSON summary:")
        print(selected.to_string(index=False))
        raise RuntimeError("No usable labeled frames found from v2 JSON files.")

    label_order = make_label_order(
        allowed_labels=allowed_labels,
        labels_seen=labeled["new_l2_label"].astype(str).unique().tolist(),
        allow_unknown_labels=bool(args.allow_unknown_labels),
    )

    labeled = labeled[labeled["new_l2_label"].isin(label_order)].copy()

    if labeled.empty:
        raise RuntimeError("All labeled frames were filtered out after label ordering.")

    case_folds = assign_new_case_folds(
        labeled=labeled,
        n_folds=int(args.n_folds),
        random_seed=int(args.random_seed),
        strategy=args.split_strategy,
    )

    write_dataset_lists(labeled, label_order, case_folds, out_level_dir, args.overwrite)

    print("[DONE] New Level-2 dataset prepared from v2 JSON files.")
    print(f"[INFO] usable labeled frames={len(labeled)} cases={labeled['case_id'].nunique()} labels={len(label_order)}")
    print("[INFO] labels:")
    for i, lab in enumerate(label_order):
        print(f"  {i}: {lab}")

    print("[INFO] selected v2 JSON summary:")
    print(selected[[
        "case_id",
        "found",
        "n_candidate_segments",
        "new_l2_segments_allowed",
        "new_l2_segments_excluded",
        "new_l2_labels_allowed",
        "all_labels_seen",
        "selected_annotation_json",
    ]].to_string(index=False))

    print("[INFO] fold summary:")
    print(pd.read_csv(out_level_dir / "new_l2_fold_summary.csv").to_string(index=False))

    print("[INFO] key audits:")
    print(f"  - {selected_path}")
    print(f"  - {all_path}")
    print(f"  - {included_path}")
    print(f"  - {excluded_path}")

    print("[INFO] key outputs:")
    print(f"  - {out_level_dir / 'new_l2_frame_labels_for_model.csv'}")
    print(f"  - {out_level_dir / 'new_l2_label_names.csv'}")
    print(f"  - {out_level_dir / 'new_l2_case_folds.csv'}")
    print(f"  - {out_level_dir / 'new_l2_dataset_class_summary.csv'}")
    print(f"  - {out_level_dir / 'new_l2_fold_summary.csv'}")
    print(f"  - {out_level_dir / 'new_l2_fold_class_summary.csv'}")


if __name__ == "__main__":
    main()
