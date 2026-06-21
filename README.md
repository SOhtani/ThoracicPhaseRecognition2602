# ThoracicPhaseRecognition2602

Artificial intelligence-based surgical phase recognition and workflow-complexity analysis for robot-assisted anatomical lung resection.

This repository contains code and documentation used to prepare, train, evaluate, and analyze a multi-level surgical phase-recognition pipeline for anatomical lung resection. It builds on the overall scheme of [`ThoracicPhaseRecognition2601`](https://github.com/SOhtani/ThoracicPhaseRecognition2601): frame-level surgical image classification, manual annotation, case-level cross-validation, mmpretrain-based model training, and fold-wise evaluation.

The current repository extends that previous wedge-resection-oriented pipeline to robot-assisted anatomical lung resection, with a multi-level label taxonomy, workspace-based Level-2 labels, stratified evaluation, and case-level workflow-complexity analysis.

## Repository status

This repository is intended for code release and reproducibility documentation. It does **not** include clinical videos, extracted image frames, patient metadata, annotation files, model checkpoints, prediction files, or other data derived from clinical records.

## Study design summary

- Procedure: robot-assisted anatomical lung resection
- Data source: surgical videos with manual annotation
- Sampling: 1 frame per second
- Cross-validation: case-level 5-fold cross-validation
- Model family: pretrained transformer-based image-classification model using an mmpretrain-style training pipeline
- Primary classification metric: macro F1
- Case-level analysis: phase transition count, phase/workspace duration, and uncertain-phase proportion correlated with operative time

## Label taxonomy

### Level 0

- `OutsideBody`
- `InsideBody`

### Level 1

- `PortDockExplore`
- `SpecimenPathway`
- `LeakHemostasis`
- `DrainClosure`

### Level 2

Workspace-based detailed process labels:

- `inferior_hilum`
- `posterior_hilum`
- `superior_hilum`
- `anterior_hilum`
- `fissure`
- `uncertain`

Mapping rules used for the final 6-class Level-2 taxonomy:

- `subcarinal` -> `posterior_hilum`
- `superior_mediastinum` -> `superior_hilum`
- `uncertain` indicates target Level-1 frames without a mapped Level-2 workspace label.

## Main results reproduced by this pipeline

### Classification performance

| Level | Task | Frames | Accuracy | Fold-mean macro F1 |
|---|---:|---:|---:|---:|
| Level 0 | extracorporeal / intracorporeal scene recognition | 587,194 | 0.989 | 0.984 |
| Level 1 | basic surgical phase recognition | 470,375 | 0.946 | 0.906 |
| Level 2 | detailed workspace phase recognition | 439,941 | 0.787 | 0.716 |

For Level 2, fold-mean macro F1 was 0.716. Macro F1 calculated from the pooled confusion matrix was 0.719.

### Case-level workflow-complexity analysis

| Metric | Label source | Spearman rho with operative time | P value |
|---|---|---:|---:|
| Phase transition count | AI-predicted | 0.735 | <0.001 |
| Phase transition count | Ground-truth | 0.562 | <0.001 |
| Time spent in fissure phase | AI-predicted | 0.606 | <0.001 |
| Time spent in fissure phase | Ground-truth | 0.601 | <0.001 |
| Proportion of uncertain phase | AI-predicted | 0.358 | 0.0047 |
| Proportion of uncertain phase | Ground-truth | 0.257 | 0.046 |

## Directory structure

```text
ThoracicPhaseRecognition2602/
  README.md
  .gitignore
  requirements.txt
  environment.yml

  configs/
    README.md

  scripts/
    00_audit_inputs.py
    01_prepare_level2_workspace_dataset.py
    02_train_all_levels.sh
    03_evaluate_workflow_predictions.py
    04_summarize_cv_metrics.py
    05_case_workflow_outcome_correlation.py
    06_plot_confusion_metric_figures.py

  docs/
    comparison_with_ThoracicPhaseRecognition2601.md
    data_requirements.md
    label_taxonomy.md
    reproducibility_notes.md

  examples/
    input_schema/
    output_schema/

  results/
    README.md
```

## Pipeline overview

### 1. Audit annotations and inputs

```bash
python scripts/00_audit_inputs.py \
  --annotation-root /path/to/annotations \
  --case-pattern 'CASE*_annotation_v2.json' \
  --out-csv outputs/input_audit.csv
```

### 2. Prepare Level-2 workspace dataset

```bash
python scripts/01_prepare_level2_workspace_dataset.py \
  --annotation-root /path/to/annotations \
  --frame-csv-dir /path/to/per_frame_csvs \
  --project-root /path/to/project_root \
  --out-level-dir /path/to/project_root/outputs/new_level2_v2_6class_uncertain \
  --case-start CASE001 \
  --case-end CASE073 \
  --n-folds 5 \
  --random-seed 42 \
  --split-strategy balanced_frames \
  --include-uncertain \
  --on-overlap skip \
  --overwrite
```

### 3. Train models

Copy the actual mmpretrain config used in the study into `configs/`, then adapt:

```bash
PROJECT_ROOT=/path/to/project_root \
MMPRETRAIN_ROOT=/path/to/mmpretrain \
CONFIG=configs/swin_small_rect.py \
WORK_DIR=outputs/work_dirs \
bash scripts/02_train_all_levels.sh
```

### 4. Evaluate predictions

```bash
python scripts/03_evaluate_workflow_predictions.py \
  --pred-dir outputs/predictions \
  --out-dir outputs/workflow_prediction_evaluation \
  --compare-by id \
  --export-frame-level \
  --overwrite
```

### 5. Summarize cross-validation metrics

```bash
python scripts/04_summarize_cv_metrics.py \
  --project-root . \
  --out-dir outputs/existing_eval_summary_v2 \
  --overwrite
```

### 6. Analyze workflow metrics and operative time

```bash
python scripts/05_case_workflow_outcome_correlation.py \
  --frame-level-csv outputs/workflow_prediction_evaluation/frame_level_predictions.csv \
  --metadata-csv data/metadata_schema_aligned.csv \
  --out-dir outputs/case_workflow_outcome \
  --sample-fps 1.0
```

### 7. Plot confusion matrices and metric figures

```bash
python scripts/06_plot_confusion_metric_figures.py \
  --project-root . \
  --out-dir outputs/publication_metric_figures \
  --level2-source current \
  --overwrite
```

## Relationship to ThoracicPhaseRecognition2601

The overall image-classification framework is inherited from `ThoracicPhaseRecognition2601`: frame-level surgical image input, manual annotation, case-level cross-validation, mmpretrain-based training, and fold-wise validation. The major differences are the target procedure and downstream analysis. This repository targets robot-assisted anatomical lung resection, introduces a Level 0/1/2 hierarchy, adds workspace-based Level-2 labels, and includes stratified and case-level workflow-complexity analyses.

See [`docs/comparison_with_ThoracicPhaseRecognition2601.md`](docs/comparison_with_ThoracicPhaseRecognition2601.md) for details.

## Data privacy

The following files must not be committed:

- raw videos
- extracted frames
- annotation JSON files containing case-level clinical labels
- Excel or CSV files containing patient or case metadata
- model checkpoints
- prediction outputs derived from clinical images
- confusion matrices or per-frame outputs if they can reveal case-level information

Use the schema examples in `examples/` to document expected formats without exposing clinical data.

## Citation

Citation information will be added after abstract/manuscript submission.
