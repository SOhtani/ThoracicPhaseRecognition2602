# configs

Place mmpretrain model configuration files here.

For this study, the analysis used a Swin Transformer-family image-classification model with ImageNet pretraining and case-level 5-fold cross-validation. The exact local config used during the analysis should be copied here, for example:

- `swin_small_rect.py`

`swin_small_rect.py` reads `DATA_ROOT` (image data root) and `PRETRAINED` (ImageNet pretrained checkpoint path) from environment variables, e.g. `DATA_ROOT=/path/to/data/images PRETRAINED=/path/to/checkpoints/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth python ...`. Both can also be overridden via `--cfg-options DATA_ROOT=... PRETRAINED=...`.

Clinical data, model checkpoints, and trained weights should not be committed.
