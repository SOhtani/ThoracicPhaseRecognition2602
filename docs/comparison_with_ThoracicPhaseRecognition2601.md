# Comparison with ThoracicPhaseRecognition2601

`ThoracicPhaseRecognition2602` is based on the same overall surgical image-classification framework as `ThoracicPhaseRecognition2601`, but it extends the task from wedge/partial resection phase recognition to robot-assisted anatomical lung resection.

| Item | ThoracicPhaseRecognition2601 | ThoracicPhaseRecognition2602 |
|---|---|---|
| Target procedure | Wedge/partial lung resection-oriented phase recognition | Robot-assisted anatomical lung resection: lobectomy and segmentectomy |
| Main aim | Surgical phase recognition | Surgical phase recognition plus workflow-complexity analysis |
| Input unit | Surgical video frames | Surgical video frames |
| Annotation source | Excel/CSV-style phase annotations | `annotation_v2` JSON and frame-level label tables |
| Sampling | Frame interval / 1 fps-style extraction depending on script settings | 1 frame per second |
| Model scheme | mmpretrain image-classification pipeline | Same overall mmpretrain image-classification scheme |
| Model family | Swin Transformer-family configuration available | Pretrained transformer-based image classifier, Swin-family in the analysis |
| Cross-validation | Case-level split using teacher CSV | Case-level 5-fold cross-validation |
| Label structure | Mainly single-level phase labels such as `1`, `2`, `3`, `4`, `X` | Hierarchical Level 0 / Level 1 / Level 2 taxonomy |
| Level 0 | Outside-body handling available | Explicit `OutsideBody` / `InsideBody` classification |
| Level 1 | Basic phase labels | `PortDockExplore`, `SpecimenPathway`, `LeakHemostasis`, `DrainClosure` |
| Level 2 | Not the central task | 6-class workspace taxonomy |
| Evaluation | Fold-wise classification metrics and confusion matrices | Fold-wise, pooled, per-class, stratified, and case-level workflow analyses |
| Downstream clinical analysis | Not central | Transition count, phase duration, uncertain proportion vs operative time |

## Inherited components

- Frame-level surgical image classification
- Manual annotation as ground truth
- Case-level cross-validation to avoid frame-level leakage
- mmpretrain-style model training and validation
- Fold-wise metric calculation
- Confusion-matrix-based model review

## Major extensions in 2602

- Application to anatomical lung resection, not wedge resection
- Explicit multi-level taxonomy
- Workspace-based Level-2 labels
- Inclusion of `uncertain` as an interpretable workspace-related category
- Stratified evaluation by side, procedure, and target lobe/segment
- Case-level workflow-complexity metrics
- Correlation of workflow metrics with operative time
