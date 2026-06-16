"""Federated Transfer Learning.

Uses pretrained models (e.g. ImageNet DenseNet-121) and fine-tunes them
across federated sites. The model architecture lives in models/hfl/densenet/;
FTL is a training configuration, not a separate model.

To run FTL:
    python run_ec2.py transfer --synthetic

See scenarios/transfer_chest.yaml for the FTL experiment configuration.
"""

# Re-export from the HFL densenet model used for transfer learning
from models.hfl.densenet.server_app import server_fn, make_strategy
from models.hfl.densenet.client_app import client_fn
