#!/usr/bin/env python3
"""
Train script (PyTorch) for binary deepfake detection on preprocessed face crops.

Expected dataset layout (default):
processed/
  train/
    original/
    deepfake/  <-- this script now accepts other names like "Deepfakes" or "deepfakes"
  val/
    original/
    deepfake/

Saves:
 - models/best_model.pt
 - models/last_model.pt
 - models/hparams.json
 - TensorBoard logs in runs/
"""

import os
import json
import time
import random
from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict, Counter

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from torch.utils.tensorboard import SummaryWriter

# === provenance: uploaded project doc (use workspace path) ===
PROPOSAL_DOC = r"/mnt/data/UFID45173502_AIS_PROJECT-PROPOSAL.docx"

# --------------------
# Utilities
# --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_hparams(models_dir: Path, hparams: dict):
    ensure_dir(models_dir)
    with open(models_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)

# --------------------
# Model factory
# --------------------
def build_model(num_classes=1, pretrained=True, freeze_backbone=False):
    """
    Build a binary classifier using EfficientNet-B0 backbone (torchvision).
    Final layer is a single sigmoid output for BCE.
    """
    backbone = models.efficientnet_b0(pretrained=pretrained)

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    # Replace classifier
    # robustly get in_features whether classifier is sequential or not
    try:
        # typical EfficientNet structure: backbone.classifier = Sequential(..., Linear)
        lin = next((m for m in backbone.classifier if isinstance(m, nn.Linear)), None)
        in_features = lin.in_features if lin is not None else 1280
    except Exception:
        in_features = getattr(backbone, "in_features", 1280)

    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.25),
        nn.Linear(512, num_classes)
    )
    return backbone

# --------------------
# Dataset label remapping wrapper
# --------------------
class LabelRemapDataset(Dataset):
    """
    Wrap an ImageFolder (or any dataset returning (img, label)) and remap label
    so that 'fake_class_index' becomes 1 and all others become 0.
    """
    def __init__(self, base_dataset, fake_class_index):
        self.base = base_dataset
        self.fake_idx = int(fake_class_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, lbl = self.base[idx]
        mapped = 1 if int(lbl) == self.fake_idx else 0
        return img, mapped

# --------------------
# Train / Eval loops
# --------------------
def train_one_epoch(model, dl, optimizer, device, scaler, criterion):
    model.train()
    losses = []
    preds_all = []
    targets_all = []
    pbar = tqdm(enumerate(dl), total=len(dl), desc="train", leave=False)
    for i, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().unsqueeze(1)  # shape [B,1]

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            out = model(x)
            loss = criterion(out, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds_all.append(torch.sigmoid(out).detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())
        pbar.set_postfix(loss=float(np.mean(losses)))
    preds = np.vstack(preds_all).reshape(-1)
    targets = np.vstack(targets_all).reshape(-1)
    auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else float("nan")
    acc = accuracy_score(targets, (preds >= 0.5).astype(int))
    return float(np.mean(losses)), auc, acc

def eval_one_epoch(model, dl, device, criterion):
    """
    Returns: val_loss, auc, acc, preds_array, targets_array
    """
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dl), total=len(dl), desc="eval", leave=False)
        for i, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds_all.append(torch.sigmoid(out).cpu().numpy())
            targets_all.append(y.cpu().numpy())
            pbar.set_postfix(loss=float(np.mean(losses)))
    preds = np.vstack(preds_all).reshape(-1)
    targets = np.vstack(targets_all).reshape(-1)
    auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else float("nan")
    acc = accuracy_score(targets, (preds >= 0.5).astype(int))
    return float(np.mean(losses)), auc, acc, preds, targets

# --------------------
# Main
# --------------------
def main(args):
    set_seed(args.seed)

    project_root = Path.cwd()
    # ensure args.data_dir is a Path object
    data_dir = Path(args.data_dir)

    models_dir = Path(args.models_dir)
    ensure_dir(models_dir)
    ensure_dir(Path("runs"))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("Device:", device)

    # transforms
    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # datasets: expects processed/train and processed/val with subfolders per class
    train_root = data_dir / "train"
    val_root = data_dir / "val"

    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            f"Expected dataset at {train_root} and {val_root}. "
            "Make sure you ran preprocessing and created processed/train and processed/val with subfolders for each class."
        )

    # load with ImageFolder (classes are taken from folder names)
    train_base = datasets.ImageFolder(root=str(train_root), transform=train_tfms)
    val_base = datasets.ImageFolder(root=str(val_root), transform=val_tfms)

    # detect which class name corresponds to fake (case-insensitive substring match)
    def find_fake_index(class_list):
        for i, cname in enumerate(class_list):
            if "fake" in cname.lower():
                return i
        # also try 'deep' heuristic
        for i, cname in enumerate(class_list):
            if "deep" in cname.lower():
                return i
        return None

    print("Detected train classes:", train_base.classes)
    print("Detected val classes:", val_base.classes)
    fake_idx_train = find_fake_index(train_base.classes)
    fake_idx_val = find_fake_index(val_base.classes)

    # require that at least one split has a detectable fake class
    fake_idx = fake_idx_train if fake_idx_train is not None else fake_idx_val
    if fake_idx is None:
        raise FileNotFoundError(
            "Could not detect a 'fake' class folder. Found classes:\n"
            f"train: {train_base.classes}\nval: {val_base.classes}\n\n"
            "Rename your fake-class folders to include the word 'fake' (e.g. 'deepfake' or 'Deepfakes'), "
            "or reorganize into processed/train/<class_name>/ and processed/val/<class_name>."
        )

    # Wrap datasets so labels are 1 for fake and 0 for all others
    train_ds = LabelRemapDataset(train_base, fake_idx)
    val_ds = LabelRemapDataset(val_base, fake_idx)

    # Print class distribution (original mapping) for visibility
    train_counts = Counter(train_base.targets)
    val_counts = Counter(val_base.targets)
    train_class_names = train_base.classes
    val_class_names = val_base.classes
    print("Train class counts (original indices):")
    for idx, cnt in sorted(train_counts.items()):
        print(f"  {train_class_names[idx]}: {cnt}")
    print("Val class counts (original indices):")
    for idx, cnt in sorted(val_counts.items()):
        print(f"  {val_class_names[idx]}: {cnt}")

    # Determine pos/neg counts after remap
    # pos_count = number of samples mapped to 1 (fake)
    pos_count = sum(1 for t in train_base.targets if int(t) == int(fake_idx))
    neg_count = len(train_base.targets) - pos_count
    print(f"Remapped training counts -> fake (1): {pos_count}, other(0): {neg_count}")

    # Create sampler if requested (helps with heavy imbalance)
    train_loader = None
    if args.use_sampler:
        # compute class sample counts over remapped labels
        if pos_count == 0:
            print("Warning: pos_count=0, cannot build sampler. Falling back to simple shuffle.")
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        else:
            # weight inversely proportional to class frequency
            target_list = [1 if int(t)==int(fake_idx) else 0 for t in train_base.targets]
            class_sample_count = np.array([len(np.where(np.array(target_list) == t)[0]) for t in [0,1]])
            weights = 1.0 / class_sample_count
            samples_weights = np.array([weights[t] for t in target_list])
            sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
            print("Using WeightedRandomSampler to balance classes in training.")

    # default train_loader if not created above
    if train_loader is None:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=1, pretrained=True, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # optionally wrap for DataParallel if >1 GPU (simple)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.data_parallel:
        model = nn.DataParallel(model)

    # optimizer / loss / scheduler
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # pos weight for BCE (helps with class imbalance). Only set if requested and positive samples exist
    if args.use_pos_weight and pos_count > 0:
        # pos_weight should be ratio of negative_count/positive_count
        pos_w = float(max(1.0, (neg_count / pos_count))) if pos_count > 0 else 1.0
        pos_weight = torch.tensor([pos_w], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_w:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        if args.use_pos_weight:
            print("Warning: pos_count==0, skipping pos_weight application.")

    scaler = GradScaler() if args.amp else None

    # tensorboard
    tb_writer = SummaryWriter(log_dir=f"runs/run_{int(time.time())}")

    best_auc = -1.0
    epoch_start = 1

    # copy-run-metadata
    hparams = OrderedDict([
        ("data_dir", str(data_dir)),
        ("models_dir", str(models_dir)),
        ("img_size", args.img_size),
        ("batch_size", args.batch_size),
        ("epochs", args.epochs),
        ("lr", args.lr),
        ("weight_decay", args.weight_decay),
        ("amp", args.amp),
        ("freeze_backbone", args.freeze_backbone),
        ("use_sampler", args.use_sampler),
        ("use_pos_weight", args.use_pos_weight),
        ("seed", args.seed),
        ("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())),
        ("proposal_doc", PROPOSAL_DOC)
    ])

    print("HParams:")
    print(json.dumps(hparams, indent=2))

    # training loop
    last_val_preds = None
    last_val_targets = None
    for epoch in range(epoch_start, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        val_loss, val_auc, val_acc, val_preds, val_targets = eval_one_epoch(model, val_loader, device, criterion)

        # keep last epoch preds for thresholding diagnostics
        last_val_preds = val_preds
        last_val_targets = val_targets

        # scheduler uses validation AUC as signal (maximize)
        scheduler.step(val_auc if not np.isnan(val_auc) else val_loss)

        # tensorboard logs
        tb_writer.add_scalar("loss/train", train_loss, epoch)
        tb_writer.add_scalar("loss/val", val_loss, epoch)
        tb_writer.add_scalar("auc/train", float(train_auc), epoch)
        tb_writer.add_scalar("auc/val", float(val_auc), epoch)
        tb_writer.add_scalar("acc/train", float(train_acc), epoch)
        tb_writer.add_scalar("acc/val", float(val_acc), epoch)
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"train loss {train_loss:.4f} auc {train_auc:.4f} acc {train_acc:.4f}")
        print(f"val   loss {val_loss:.4f} auc {val_auc:.4f} acc {val_acc:.4f}")

        # checkpointing
        last_path = models_dir / "last_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": float(val_auc),
            "val_loss": float(val_loss),
            "hparams": hparams
        }, last_path)

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_path = models_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": float(val_auc),
                "val_loss": float(val_loss),
                "hparams": hparams
            }, best_path)
            print(f"[CHECKPOINT] saved new best model to {best_path} (AUC {best_auc:.4f})")

    # final save hparams & summary
    # compute recommended threshold from last validation preds (Youden's J)
    recommended_threshold = None
    try:
        if last_val_preds is not None and last_val_targets is not None and len(np.unique(last_val_targets)) > 1:
            fpr, tpr, th = roc_curve(last_val_targets, last_val_preds)
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            recommended_threshold = float(th[best_idx])
            print(f"Recommended threshold from val ROC (Youden's J): {recommended_threshold:.4f}")
            hparams["recommended_threshold"] = recommended_threshold
    except Exception as e:
        print("Warning: could not compute recommended threshold:", e)

    hparams["best_val_auc"] = float(best_auc)
    hparams["completed_epochs"] = args.epochs
    save_hparams(models_dir, hparams)
    tb_writer.close()
    print("Training complete. Artifacts saved to:", models_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="processed", help="root processed dataset dir (train/val)")
    parser.add_argument("--models-dir", type=str, default="models", help="where to save models and hparams")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="enable mixed precision")
    parser.add_argument("--use-cuda", action="store_true", help="use GPU if available")
    parser.add_argument("--freeze-backbone", action="store_true", help="freeze backbone weights")
    parser.add_argument("--data-parallel", action="store_true", help="wrap model in DataParallel when multiple GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-sampler", action="store_true", help="use WeightedRandomSampler to balance training batches")
    parser.add_argument("--use-pos-weight", action="store_true", help="set BCEWithLogitsLoss pos_weight from class counts (helps imbalance)")
    args = parser.parse_args()
    main(args)
