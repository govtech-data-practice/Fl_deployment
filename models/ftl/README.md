# Federated Transfer Learning (FTL)

FTL uses a pretrained model (e.g. ImageNet DenseNet-121) and fine-tunes it across federated sites. Each site fine-tunes on its local data; the server aggregates the fine-tuned weights.

## How It Works

```
                  Pretrained Model (e.g. ImageNet DenseNet-121)
                         |
         +---------------+---------------+
         |               |               |
    Site A           Site B           Site C
    [fine-tune on    [fine-tune on    [fine-tune on
     local X-rays]    local X-rays]    local X-rays]
         |               |               |
         +---- FL Server aggregates fine-tuned weights ----+
```

## Implementation

FTL is a **training configuration**, not a separate model architecture. The model lives in `models/hfl/densenet/` and is used with transfer learning settings:

- Pretrained ImageNet weights loaded at initialisation
- Only classifier head is trained (feature extractor frozen initially)
- Gradual unfreezing during later rounds

## Usage

```bash
python run_ec2.py transfer --synthetic
```

See `scenarios/transfer_chest.yaml` for the experiment definition.
