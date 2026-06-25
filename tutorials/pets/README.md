# PET Tools — Hands-on Tutorials

Focused tutorials for each Privacy-Enhancing Technology (PET) in the `fl_pets/` toolkit.
These are tool-level guides — for the theory and FL integration, see the [main tutorials](../README.md).

## Prerequisites

- **Python 3.10+** and `pip install -e ".[dev]"`
- Each tutorial lists additional library requirements

## Tutorials

| PET | Tutorial | Library | Install |
|-----|----------|---------|---------|
| Private Set Alignment | [PSA: Entity Alignment](psa-entity-alignment.ipynb) | anonlink + clkhash (Data61) | `pip install anonlink clkhash` |
| Differential Privacy | [DP: Gradient Privacy](dp-gradient-privacy.ipynb) | Opacus (Meta) | `pip install opacus` |
| Secure Aggregation | [SecAgg: Update Masking](secagg-update-masking.ipynb) | Flower SecAgg+ | `pip install flwr` |
| Secure Inference | [HE vs MPC Comparison](secure-inference.ipynb) | TenSEAL + CrypTen | `pip install tenseal` |

## PET Lifecycle

```
Pre-training          During training         Inference            Post-training
+-----------+         +------+  +------+      +----+  +-----+     +---------+
| PSA       |   -->   | DP   |  |SecAgg|  --> | HE |  | MPC | --> | Privacy |
| (align)   |         | (noise)  (mask) |     |(enc)  (split)|    | attacks |
+-----------+         +------+  +------+      +----+  +-----+     +---------+
```

## Related

- [Tutorial 4: Differential Privacy](../intermediate/04-differential-privacy.ipynb) — DP in the FL training loop
- [Tutorial 5: Secure Aggregation](../intermediate/05-secure-aggregation.ipynb) — SecAgg with Flower
- [Tutorial 7: Privacy Attack Testing](../intermediate/07-privacy-attacks.ipynb) — validate PET effectiveness
- [Tutorial 10: Vertical FL & PSA](../advanced/10-vertical-fl.md) — PSA in the VFL pipeline
- [PET Reference](../reference/PET_Reference.md) — full technical reference
