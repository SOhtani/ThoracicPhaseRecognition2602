# configs/swin_small_rect.py
# 汎用（Swin-S, 長辺横, 224x448 などの長方形）
import os

_base_ = []

# ===== 外部から --cfg-options で上書き可能な変数 =====
# DATA_ROOT / PRETRAINED はローカル環境固有のパスなので、環境変数で渡す
# （未設定時はプレースホルダ。run_all_levels.py から --cfg-options で注入も可能）
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data/images")
NUM_CLASSES = 2
IMG_H = 224
IMG_W = 448
PRETRAINED = os.environ.get(
    "PRETRAINED",
    "/path/to/pretrained/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth",
)

# ===== モデル =====
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=(IMG_H, IMG_W),
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=PRETRAINED),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=NUM_CLASSES,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

# ===== パイプライン（H, W の順に注意）=====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',     scale=(IMG_W, IMG_H)),
    dict(type='RandomCrop', crop_size=(IMG_H, IMG_W)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',     scale=(IMG_W, IMG_H)),
    dict(type='CenterCrop', crop_size=(IMG_H, IMG_W)),
    dict(type='PackInputs'),
]

# ===== データローダ（ann_file は外から run_all_levels.py が指定）=====
train_dataloader = dict(
    batch_size=64, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=2,
    dataset=dict(
        type='CustomDataset',
        data_root=DATA_ROOT,
        ann_file='',  # 空→ CLI から --cfg-options で注入
        data_prefix='',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=64, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=2,
    dataset=dict(
        type='CustomDataset',
        data_root=DATA_ROOT,
        ann_file='',
        data_prefix='',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='Accuracy', topk=(1,))
test_evaluator = dict(type='Accuracy', topk=(1,))

optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05)
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=50, by_epoch=True, begin=0, end=50),
]

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_scope = 'mmpretrain'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='accuracy/top1'),
    logger=dict(type='LoggerHook', interval=500),
)
custom_hooks = [
    dict(type='EarlyStoppingHook', patience=3, min_delta=5e-4, monitor='accuracy/top1', rule='greater')
]

env_cfg = dict(cudnn_benchmark=True)
log_level = 'INFO'
