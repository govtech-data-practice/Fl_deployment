"""Chest X-ray FL ClientApp (DenseNet-121)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

import flwr as fl
from flwr.client import Client
from flwr.clientapp import ClientApp
from flwr.common import Context

from fl_common import build_trainable_to_state_map, secagg_mask_parameters, clip_and_noise

logger = logging.getLogger("chest.client")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 14


class ChestXrayDenseNet121(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, dropout_rate=0.2):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = densenet121(weights=weights)
        nf = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(nf, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.base_model(x)


# --- Synthetic dataset (CPU testing) ---
class SyntheticDataset(Dataset):
    def __init__(self, n, seed=42):
        rng = np.random.RandomState(seed)
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 3, 224, 224, generator=g)
        self.y = torch.from_numpy((rng.rand(n, NUM_CLASSES) > 0.8).astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# --- Real dataset ---
class RealDataset(Dataset):
    def __init__(self, data_dict, transform, path):
        self.imgs = data_dict["images"]
        self.labels = data_dict["labels"]
        self.transform = transform
        self.path = path
        self._c = {}

    def _find(self, name):
        if name in self._c:
            return self._c[name]
        for d in ["images"] + [f"images_{i:03d}/images" for i in range(1, 13)]:
            p = os.path.join(self.path, d, name)
            if os.path.exists(p):
                self._c[name] = p
                return p
        return None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from PIL import Image
        p = self._find(self.imgs[i])
        try:
            img = Image.open(p).convert("RGB") if p else Image.new("RGB", (224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, torch.FloatTensor(self.labels[i])


# --- Data loading ---
_cache = {"key": None, "train": None, "val": None}


_global_val_cache = {"loaded": False, "loader": None}
_held_out_patients = None  # patients reserved for test — never used for training


def _get_held_out_patients(csv_path):
    """Hold out 20% of patients globally BEFORE any FL partitioning."""
    global _held_out_patients
    if _held_out_patients is not None:
        return _held_out_patients
    import pandas as pd
    df = pd.read_csv(csv_path)
    patients = df["Patient ID"].unique()
    rng = np.random.RandomState(7777)  # fixed seed, independent from FL partition seed
    rng.shuffle(patients)
    n_test = int(len(patients) * 0.2)
    _held_out_patients = set(patients[:n_test])
    logger.info(f"Held out {n_test} patients for global test (never in training)")
    return _held_out_patients


def _load(pid, n_clients, ptype, alpha, synthetic, dataset_path, csv_path):
    key = f"{ptype}_{alpha}_{pid}_{synthetic}"
    if _cache["key"] == key and _cache["train"]:
        return _cache["train"], _cache["val"]

    if synthetic:
        per = 500 // n_clients
        train_n, val_n = int(per * 0.8), max(per - int(per * 0.8), 20)
        # All clients share same val set for fair comparison
        train = DataLoader(SyntheticDataset(train_n, seed=42 + pid), batch_size=8, shuffle=True)
        val = DataLoader(SyntheticDataset(val_n, seed=9999), batch_size=8)
    else:
        import pandas as pd
        import torchvision.transforms as T
        from tasks.hfl.chest_xray.data import partition_data_dynamic, CLASSES

        csv = csv_path or os.path.join(dataset_path, "Data_Entry_2017.csv")

        # Step 1: Get held-out test patients (global, fixed)
        test_patients = _get_held_out_patients(csv)

        # Step 2: Partition only TRAIN patients across FL clients
        # partition_data_dynamic will be called on the full CSV, but we filter
        # the result to exclude test patients
        method = "iid" if ptype == "iid" else "patient"
        data = partition_data_dynamic(csv_path=csv, client_id=pid, num_clients=n_clients,
                                      method=method, alpha=alpha, seed=42)

        # Filter out test patients from training data
        df = pd.read_csv(csv)
        img_to_patient = dict(zip(df["Image Index"], df["Patient ID"]))
        train_imgs = [img for img in data["train"]["images"]
                      if img_to_patient.get(img, "") not in test_patients]
        # Rebuild labels for filtered images
        train_idx = [i for i, img in enumerate(data["train"]["images"])
                     if img_to_patient.get(img, "") not in test_patients]
        train_labels = data["train"]["labels"][train_idx] if len(train_idx) > 0 else data["train"]["labels"][:0]

        tt = T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
        tv = T.Compose([T.Resize(256), T.CenterCrop(224),
                        T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

        train = DataLoader(
            RealDataset({"images": np.array(train_imgs), "labels": train_labels}, tt, dataset_path),
            batch_size=16, shuffle=True, num_workers=2, pin_memory=False)

        # Step 3: Global test set from held-out patients (same for ALL clients)
        if not _global_val_cache["loaded"]:
            # Load test images from held-out patients
            df["labels_list"] = df["Finding Labels"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
            for c in CLASSES:
                if c not in df.columns:
                    df[c] = df["labels_list"].apply(lambda x: 1 if c in x else 0)
            has_finding = df[CLASSES].sum(axis=1) > 0
            test_df = df[has_finding & df["Patient ID"].isin(test_patients)]
            # Cap at 2000 for speed
            if len(test_df) > 2000:
                test_df = test_df.sample(2000, random_state=7777)
            val_imgs = test_df["Image Index"].values
            val_labels = test_df[CLASSES].values.astype(np.float32)
            _global_val_cache["loader"] = DataLoader(
                RealDataset({"images": val_imgs, "labels": val_labels}, tv, dataset_path),
                batch_size=32, num_workers=2, pin_memory=False)
            _global_val_cache["loaded"] = True
            logger.info(f"Global test set: {len(val_imgs)} samples from {len(test_patients)} held-out patients")
        val = _global_val_cache["loader"]

    _cache.update(key=key, train=train, val=val)
    logger.info(f"Loaded: {len(train.dataset)} train, {len(val.dataset)} val (synthetic={synthetic})")
    return train, val


class ChestClient(fl.client.NumPyClient):
    def __init__(self, pid, n_clients, synthetic, dataset_path, csv_path):
        self.pid, self.n_clients = pid, n_clients
        self.synthetic, self.dataset_path, self.csv_path = synthetic, dataset_path, csv_path
        self.model = ChestXrayDenseNet121(pretrained=not synthetic).to(DEVICE)
        self.criterion = nn.BCELoss()
        self._tmap = build_trainable_to_state_map(self.model)
        self.global_params = None
        self.client_control = self.server_control = self.prev_params = None

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = dict(zip(self.model.state_dict().keys(), params))
        clamped = {}
        for k, v in sd.items():
            t = torch.tensor(np.array(v), device=DEVICE)
            if not torch.isfinite(t).all():
                t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
            clamped[k] = t
        self.model.load_state_dict(clamped, strict=True)

    def fit(self, parameters, config):
        strategy = config.get("strategy", "fedavg")
        lr = float(config.get("learning_rate", 0.0001))
        epochs = int(config.get("local_epochs", 1))
        ptype = config.get("partition_type", "iid")
        alpha = float(config.get("alpha", 100.0))
        prox_mu = float(config["proximal_mu"]) if "proximal_mu" in config else None
        secagg_seed = int(config["secagg_round_seed"]) if "secagg_round_seed" in config else None
        secagg_n = int(config.get("secagg_num_clients", self.n_clients))
        dp_mode = config.get("dp_mode")
        dp_noise = float(config.get("dp_noise_multiplier", 1.0))
        dp_clip = float(config.get("dp_max_grad_norm", 1.0))
        dp_seed = int(config["dp_seed"]) if "dp_seed" in config else None

        train, _ = _load(self.pid, self.n_clients, ptype, alpha,
                         self.synthetic, self.dataset_path, self.csv_path)
        self.prev_params = [p.copy() for p in parameters]
        self.set_parameters(parameters)

        if strategy == "scaffold":
            if not self.client_control:
                self.client_control = [np.zeros_like(p) for p in parameters]
            if not self.server_control:
                self.server_control = [np.zeros_like(p) for p in parameters]
        if prox_mu is not None:
            self.global_params = [p.clone().detach() for p in self.model.parameters()]

        opt = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        total_loss, nb = 0.0, 0

        for _ in range(epochs):
            for imgs, labels in train:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                loss = self.criterion(self.model(imgs).clamp(1e-7, 1-1e-7), labels)
                if prox_mu and self.global_params:
                    loss = loss + (prox_mu / 2) * sum(
                        ((lp - gp)**2).sum() for lp, gp in zip(self.model.parameters(), self.global_params))
                loss.backward()
                if strategy == "scaffold" and self.server_control:
                    with torch.no_grad():
                        for ti, p in enumerate(self.model.parameters()):
                            if p.grad is not None:
                                si = self._tmap[ti]
                                p.grad.add_(torch.tensor(
                                    self.server_control[si] - self.client_control[si],
                                    device=DEVICE, dtype=p.grad.dtype))
                opt.step()
                total_loss += loss.item(); nb += 1

        new_p = self.get_parameters({})
        # SCAFFOLD: update control variates (only trainable params, skip buffers)
        if strategy == "scaffold" and self.prev_params is not None:
            K = epochs * len(train)
            trainable_indices = set(self._tmap.values())
            for i in range(len(self.client_control)):
                if i not in trainable_indices:
                    continue
                d = (self.prev_params[i] - new_p[i]) / (K * lr)
                self.client_control[i] = self.client_control[i] - self.server_control[i] + d
        if strategy == "secagg" and secagg_seed is not None:
            new_p = secagg_mask_parameters(new_p, self.pid, secagg_n, int(secagg_seed))

        # Local DP: client clips update and adds noise before sending
        if strategy == "dp" and dp_mode == "local":
            seed = int(dp_seed) + self.pid if dp_seed else None
            new_p = clip_and_noise(self.prev_params, new_p, dp_clip, dp_noise, seed)

        return new_p, len(train.dataset), {"loss": total_loss / max(nb, 1)}

    def evaluate(self, parameters, config):
        ptype = config.get("partition_type", "iid")
        alpha = float(config.get("alpha", 100.0))
        _, val = _load(self.pid, self.n_clients, ptype, alpha,
                       self.synthetic, self.dataset_path, self.csv_path)
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in val:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = self.model(imgs).clamp(1e-7, 1-1e-7)
                total_loss += self.criterion(out, labels).item()
                preds_all.append(out.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        auc = 0.5  # default = random
        if preds_all:
            try:
                p = np.vstack(preds_all)
                l = np.vstack(labels_all)
                # Per-class AUC, skip classes with single label value
                class_aucs = []
                for c in range(p.shape[1]):
                    if len(np.unique(l[:, c])) > 1:
                        class_aucs.append(roc_auc_score(l[:, c], p[:, c]))
                auc = float(np.mean(class_aucs)) if class_aucs else 0.5
            except (ValueError, IndexError):
                auc = 0.5
        return total_loss / max(len(val), 1), len(val.dataset), {"auc": auc, "loss": total_loss / max(len(val), 1)}


def client_fn(context: Context) -> Client:
    nc = context.node_config
    pid = int(nc.get("partition-id", 0))
    n = int(nc.get("num-clients", 0) or nc.get("num-partitions", 0) or os.environ.get("NUM_CLIENTS", 2))
    syn = str(nc.get("synthetic", "") or os.environ.get("SYNTHETIC", "1")).lower() in ("1", "true", "yes")
    dp = str(nc.get("data-path", "") or os.environ.get("DATASET_PATH", "/data"))
    csv = str(nc.get("csv-path", "") or os.environ.get("CSV_PATH", "")) or None
    return ChestClient(pid, n, syn, dp, csv).to_client()


app = ClientApp(client_fn=client_fn)
