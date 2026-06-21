#!/usr/bin/env bash
set -euo pipefail

# Example training launcher. Adapt paths and config names to your local mmpretrain environment.
# This repository does not include clinical images, annotations, or trained weights.

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
MMPRETRAIN_ROOT=${MMPRETRAIN_ROOT:-/mmpretrain}
CONFIG=${CONFIG:-configs/swin_small_rect.py}
WORK_DIR=${WORK_DIR:-outputs/work_dirs}

for LEVEL in level0 level1 level2; do
  for FOLD in 0 1 2 3 4; do
    echo "Training ${LEVEL} fold ${FOLD}"
    python "${MMPRETRAIN_ROOT}/tools/train.py" "${CONFIG}" \
      --work-dir "${WORK_DIR}/${LEVEL}/fold${FOLD}" \
      --cfg-options \
        train_dataloader.dataset.ann_file="${PROJECT_ROOT}/data/prepared/folds_png/${LEVEL}/cv${FOLD}/_normalized/train_cv${FOLD}.txt" \
        val_dataloader.dataset.ann_file="${PROJECT_ROOT}/data/prepared/folds_png/${LEVEL}/cv${FOLD}/_normalized/test_cv${FOLD}.txt"
  done
done
