#!/usr/bin/env python3
"""Centralized baseline: DenseNet-121 on ALL chest X-ray data (no federation)."""
import sys, os, time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from torchvision.models import densenet121, DenseNet121_Weights
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("central")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV = os.environ.get("CSV_PATH", "/data/chest_xray_real/archive/Data_Entry_2017.csv")
DATA = os.environ.get("DATASET_PATH", "/data/chest_xray_real/archive")
NC = 14
EPOCHS = 10
BS = 32
LR = 0.0001

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]


class XrayDS(Dataset):
    def __init__(self, imgs, labs, tf, path):
        self.imgs, self.labs, self.tf, self.path = imgs, labs, tf, path
        self._c = {}

    def _find(self, n):
        if n in self._c:
            return self._c[n]
        for d in ["images"] + [f"images_{i:03d}/images" for i in range(1, 13)]:
            p = os.path.join(self.path, d, n)
            if os.path.exists(p):
                self._c[n] = p
                return p
        return None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self._find(self.imgs[i])
        try:
            img = Image.open(p).convert("RGB") if p else Image.new("RGB", (224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tf(img), torch.FloatTensor(self.labs[i])


def load_data():
    import pandas as pd
    df = pd.read_csv(CSV)
    df["labels"] = df["Finding Labels"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    for c in CLASSES:
        df[c] = df["labels"].apply(lambda x: 1 if c in x else 0)
    has = df[CLASSES].sum(axis=1) > 0
    df = df[has].copy()

    np.random.seed(42)
    patients = df["Patient ID"].unique()
    np.random.shuffle(patients)
    val_n = int(len(patients) * 0.2)
    val_p = set(patients[:val_n])

    train_df = df[~df["Patient ID"].isin(val_p)]
    val_df = df[df["Patient ID"].isin(val_p)]
    if len(val_df) > 2000:
        val_df = val_df.sample(2000, random_state=999)

    def to_dict(d):
        return {"images": d["Image Index"].values, "labels": d[CLASSES].values.astype(np.float32)}

    return to_dict(train_df), to_dict(val_df)


def main():
    SEP = "=" * 60
    logger.info(SEP)
    logger.info("CENTRALIZED BASELINE: DenseNet-121 on ALL chest X-ray data")
    logger.info("Device: " + DEVICE)
    if DEVICE == "cuda":
        logger.info("GPU: " + torch.cuda.get_device_name(0))
    logger.info(SEP)

    train_d, val_d = load_data()
    logger.info("Train: %d, Val: %d" % (len(train_d["images"]), len(val_d["images"])))

    tt = T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(),
                     T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    tv = T.Compose([T.Resize(256), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

    train_loader = DataLoader(XrayDS(train_d["images"], train_d["labels"], tt, DATA),
                              batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(XrayDS(val_d["images"], val_d["labels"], tv, DATA),
                            batch_size=BS, num_workers=2, pin_memory=True)

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    nf = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, NC), nn.Sigmoid())
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCELoss()

    logger.info("Training %d epochs, bs=%d, lr=%s" % (EPOCHS, BS, LR))
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(imgs), labs)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(DEVICE)
                preds.append(model(imgs).cpu().numpy())
                labels.append(labs.numpy())
        p, l = np.vstack(preds), np.vstack(labels)
        class_aucs = []
        for c in range(NC):
            if len(np.unique(l[:, c])) > 1:
                class_aucs.append(roc_auc_score(l[:, c], p[:, c]))
        auc = np.mean(class_aucs) if class_aucs else 0.5

        logger.info("  Epoch %d/%d: loss=%.4f, AUC=%.4f (%ds)" % (ep, EPOCHS, ep_loss / nb, auc, time.time() - t0))

    logger.info("")
    logger.info(SEP)
    logger.info("CENTRALIZED RESULT: AUC=%.4f" % auc)
    logger.info("Total time: %ds" % (time.time() - t0))
    logger.info(SEP)


if __name__ == "__main__":
    main()
