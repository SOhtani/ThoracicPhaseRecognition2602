# Reproducibility notes

## Case-level split

All training/validation/test splits should be performed at the case level. Frames from the same surgical case must not appear in both training and validation/test data.

## Sampling rate

The study used 1 frame per second for frame-level sampling. Case-level duration estimates in the workflow analysis assume `--sample-fps 1.0` unless a `time_sec` column is provided.

## Level-2 taxonomy

The final Level-2 analysis uses the 6-class workspace taxonomy:

```text
inferior_hilum
posterior_hilum
superior_hilum
anterior_hilum
fissure
uncertain
```

`subcarinal` and `superior_mediastinum` were absorbed into `posterior_hilum` and `superior_hilum`, respectively.

## Known metadata corrections used in the study

The following case-level metadata corrections were applied before abstract-level reporting and stratified interpretation:

- `CASE020`: left upper lobectomy / left upper lobe
- `CASE034`: left S6
- `CASE043`: metadata procedure typo corrected from partial resection to lobectomy/right upper lobe for cohort summary

Do not commit original metadata spreadsheets. If such corrections are needed for reproducibility, document them in a de-identified correction log.
