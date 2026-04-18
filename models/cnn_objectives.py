# models/cnn_objectives.py
"""
Upgraded CNN objective regressors.

Architecture change from baseline:
  Baseline : 4 × [Conv → AvgPool → BN → ReLU] → Dropout → FC
  New       : ResNet-18-style encoder with:
                - Initial stem conv (7×7, stride 2)
                - 4 residual stages with BasicBlock (skip connections)
                - Global Average Pooling
                - Dropout → FC head
              Input: (B, 1, 64, 64) → scalar

Improvements:
  • Skip connections prevent vanishing gradients → better convergence
  • Larger effective receptive field → captures long-range correlations
    between lesion and background (critical for NSNR and RMSE tasks)
  • Per-objective label normalisation (z-score) for more stable training
  • Gradient clipping during training
  • Mixed precision training (if CUDA available)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.v2 as transforms
import h5py
import numpy as np
from pathlib import Path

OBJECTIVE_NAMES = ["inv_rmse", "nsnr", "inv_fwhm"]

# ─────────────────────────────────────────────────────────────────────────────
# ResNet BasicBlock
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3×3 convs with skip connection."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 style encoder (single-channel input, scalar output)
# ─────────────────────────────────────────────────────────────────────────────

class ResNetObjectiveCNN(nn.Module):
    """
    ResNet-18-like network adapted for 64×64 single-channel PET images.

    Stage dimensions for 64×64 input:
      stem     → 32×32 (stride-2 MaxPool)
      layer1   → 32×32  (no stride)
      layer2   → 16×16  (stride-2)
      layer3   →  8×8   (stride-2)
      layer4   →  4×4   (stride-2)
      GAP      →  1×1
      FC       → scalar
    """
    def __init__(self, in_channels=1, dropout=0.5):
        super().__init__()

        # Stem: adapted for small 64×64 images (smaller kernel than standard ResNet)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # → 16×16
        )

        # Residual stages
        self.layer1 = self._make_layer(16,  32,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(32,  64,  n_blocks=2, stride=2)
        self.layer3 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 128, n_blocks=2, stride=2)

        # Head
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(128, 1)

        # Weight initialisation (Kaiming)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.dropout(x.view(x.size(0), -1))
        return self.fc(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline CNN (Original Paper Architecture)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineObjectiveCNN(nn.Module):
    """
    Original 4-block architecture from the MOEAP paper.
    [Conv2d(3x3) -> AvgPool2d(2x2) -> BatchNorm2d -> ReLU] x 4
    Filter depths: 8, 16, 32, 32.
    """
    def __init__(self, in_channels=1, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


# Toggle this alias to switch between architectures
ObjectiveCNN = ResNetObjectiveCNN
# ObjectiveCNN = BaselineObjectiveCNN


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with z-score label normalisation
# ─────────────────────────────────────────────────────────────────────────────

class PETDataset(Dataset):
    def __init__(self, h5_path, split="train", objective_idx=0,
                 label_mean=None, label_std=None):
        with h5py.File(h5_path, "r") as f:
            self.images = f[f"{split}/images"][:]    # (N,1,H,W)
            raw_labels  = f[f"{split}/labels"][:, objective_idx]

        # Normalise images to [0,1] per sample
        img_max = self.images.max(axis=(1, 2, 3), keepdims=True) + 1e-8
        self.images = (self.images / img_max).astype(np.float32)

        # Z-score labels (fit stats on train set, apply to val/test)
        if label_mean is None:
            label_mean = float(raw_labels.mean())
            label_std  = float(raw_labels.std()) + 1e-8
        self.label_mean = label_mean
        self.label_std  = label_std
        self.labels = ((raw_labels - label_mean) / label_std).astype(np.float32)

        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
            ])
        else:
            self.transform = None

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return (img, torch.tensor(self.labels[idx]))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(h5_path="data/pet_dataset.h5",
                objective_idx=0,
                save_dir="models/checkpoints",
                epochs=80,
                patience=20,
                lr=3e-4,
                batch_size=64,
                weight_decay=1e-2):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    name = OBJECTIVE_NAMES[objective_idx]
    print(f"\n=== Training ResNet model for: {name} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    # Compute label stats from training set
    train_ds = PETDataset(h5_path, "train", objective_idx)
    lm, ls   = train_ds.label_mean, train_ds.label_std
    val_ds   = PETDataset(h5_path, "val",   objective_idx,
                           label_mean=lm, label_std=ls)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=(device.type == "cuda"))

    model     = ObjectiveCNN().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.HuberLoss(delta=1.0)   # robust to label outliers
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val   = float("inf")
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= len(train_ds)
        train_hist.append(tr_loss)

        # Validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    va_loss += criterion(model(imgs), lbls).item() * len(imgs)
        va_loss /= len(val_ds)
        val_hist.append(va_loss)
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({
                "state_dict":  model.state_dict(),
                "label_mean":  lm,
                "label_std":   ls,
                "objective":   name,
            }, f"{save_dir}/{name}_best.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
                  f"lr={current_lr:.2e}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    np.save(f"{save_dir}/{name}_train_hist.npy", np.array(train_hist))
    np.save(f"{save_dir}/{name}_val_hist.npy",   np.array(val_hist))
    print(f"  Best val loss: {best_val:.4f}")
    return best_val


def load_models(save_dir="models/checkpoints", device=None, ignore_fwhm=False):
    """Load trained CNN objective models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    objs = [o for o in OBJECTIVE_NAMES if not (ignore_fwhm and o == "inv_fwhm")]
    for name in objs:
        m = ObjectiveCNN().to(device)
        try:
            ckpt = torch.load(f"{save_dir}/{name}_best.pt", map_location=device)
            m.load_state_dict(ckpt["state_dict"])
        except FileNotFoundError:
            print(f"  [WARNING] Checkpoint {name}_best.pt not found. Using randomly initialized UNTRAINED model.")
        m.eval()
        models[name] = m
    return models


def evaluate_all(h5_path="data/pet_dataset.h5", save_dir="models/checkpoints"):
    """Pearson correlation on test set for each model."""
    import scipy.stats as stats
    device = torch.device("cpu")
    for idx, name in enumerate(OBJECTIVE_NAMES):
        ckpt  = torch.load(f"{save_dir}/{name}_best.pt", map_location=device)
        lm, ls = ckpt["label_mean"], ckpt["label_std"]
        ds    = PETDataset(h5_path, "test", idx, label_mean=lm, label_std=ls)
        dl    = DataLoader(ds, batch_size=128, shuffle=False)
        m     = ObjectiveCNN().to(device)
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        preds, truths = [], []
        with torch.no_grad():
            for imgs, lbls in dl:
                preds.append(m(imgs).numpy())
                truths.append(lbls.numpy())
        preds  = np.concatenate(preds)
        truths = np.concatenate(truths)
        r, p   = stats.pearsonr(preds, truths)
        rmse   = np.sqrt(np.mean((preds - truths)**2))
        print(f"{name}: r={r:.3f}, p={p:.3e}, RMSE={rmse:.4f}")


if __name__ == "__main__":
    for i in range(3):
        train_model(objective_idx=i)
    evaluate_all()