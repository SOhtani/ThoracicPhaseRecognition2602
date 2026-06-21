# Label taxonomy

## Level 0

Scene-level body-context classification.

| Label | Meaning |
|---|---|
| `OutsideBody` | Outside the thoracic operative field / extracorporeal scene |
| `InsideBody` | Intrathoracic operative scene |

## Level 1

Basic surgical process labels.

| Label | Meaning |
|---|---|
| `PortDockExplore` | Port placement, docking, and exploratory phase |
| `SpecimenPathway` | Main operative pathway toward specimen preparation/resection |
| `LeakHemostasis` | Air leak check, hemostasis, or related confirmation/manipulation |
| `DrainClosure` | Drain placement and closure-related phase |

## Level 2

Detailed anatomical workspace labels used within target Level-1 phases.

| Final label | Notes |
|---|---|
| `inferior_hilum` | Inferior hilar workspace |
| `posterior_hilum` | Posterior hilar workspace; includes mapped `subcarinal` labels |
| `superior_hilum` | Superior hilar workspace; includes mapped `superior_mediastinum` labels |
| `anterior_hilum` | Anterior hilar workspace |
| `fissure` | Fissure workspace |
| `uncertain` | Target Level-1 frame without a mapped detailed workspace label |

## Mapping rules

- `subcarinal` -> `posterior_hilum`
- `superior_mediastinum` -> `superior_hilum`
- Special or non-target labels should be excluded unless explicitly enabled for exploratory analysis.
