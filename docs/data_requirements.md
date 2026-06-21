# Data requirements

This repository does not include clinical data. The scripts expect local data prepared by the user.

## Required inputs

### Annotation files

Expected pattern:

```text
CASE001_annotation_v2.json
CASE002_annotation_v2.json
...
```

The Level-2 preparation script attempts to read several possible JSON structures, including:

- `active_annotation.segments`
- `segments`
- `layers[].segments`
- `layers[].items`
- nested dictionaries with label and start/end fields

### Frame-level CSV files

The Level-2 preparation script expects per-case frame-level tables that can be matched to annotation intervals. The exact local schema may vary; document any project-specific schema in a private, non-committed note.

### Metadata for outcome analysis

For `scripts/05_case_workflow_outcome_correlation.py`, a metadata CSV should contain at least:

```text
case_id,operative_time_min
```

Optional columns:

```text
blood_loss_g,side,procedure,target
```

## Files that should not be committed

- raw videos
- extracted frame images
- original annotation files
- original patient metadata spreadsheets
- model checkpoints
- frame-level predictions
- case-level clinical outcome tables
