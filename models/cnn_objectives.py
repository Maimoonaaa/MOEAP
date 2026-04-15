# models/cnn_objectives.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path

# ── Architecture (paper: 4 × [Conv→AvgPool→BN→ReLU] → Dropout → FC) ──────────

class ObjectiveCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.blocks = nn.Sequential(
            self._block(in_channels, 8),   # depth 8
            self._block(8,  16),            # depth 16
            self._block(16, 32),            # depth 32
            self._block(32, 32),            # depth 32
        )
        self.dropout = nn.Dropout(0.2)
        # After 4 AvgPool2×2 on 64×64 input → 4×4 feature map
        self.fc = nn.Linear(32 * 4 * 4, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PETDataset(Dataset):
    def __init__(self, h5_path, split="train", objective_idx=0):
        """objective_idx: 0=inv_RMSE, 1=NSNR, 2=inv_FWHM"""
        with h5py.File(h5_path, "r") as f:
            self.images = f[f"{split}/images"][:]   # (N,1,H,W)
            self.labels = f[f"{split}/labels"][:, objective_idx]  # (N,)
        # Normalise images to [0,1]
        self.images = self.images / (self.images.max() + 1e-8)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return (torch.tensor(self.images[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx],  dtype=torch.float32))


# ── Training loop ─────────────────────────────────────────────────────────────

OBJECTIVE_NAMES = ["inv_rmse", "nsnr", "inv_fwhm"]

def train_model(h5_path="data/pet_dataset.h5",
                objective_idx=0,
                save_dir="models/checkpoints",
                epochs=60,
                patience=15,
                lr=1e-3,
                batch_size=64):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    name = OBJECTIVE_NAMES[objective_idx]
    print(f"\n=== Training model for: {name} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PETDataset(h5_path, "train", objective_idx)
    val_ds   = PETDataset(h5_path, "val",   objective_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = ObjectiveCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                pred = model(imgs)
                val_loss += criterion(pred, lbls).item() * len(imgs)
        val_loss /= len(val_ds)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(),
                       f"{save_dir}/{name}_best.pt")
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"  Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def load_models(save_dir="models/checkpoints", device=None):
    """Load all three trained CNN objective models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for idx, name in enumerate(OBJECTIVE_NAMES):
        m = ObjectiveCNN().to(device)
        m.load_state_dict(torch.load(
            f"{save_dir}/{name}_best.pt", map_location=device))
        m.eval()
        models[name] = m
    return models


def evaluate_all(h5_path="data/pet_dataset.h5",
                 save_dir="models/checkpoints"):
    """Correlation on test set for each model."""
    import scipy.stats as stats
    device = torch.device("cpu")
    models = load_models(save_dir, device)

    for idx, name in enumerate(OBJECTIVE_NAMES):
        ds = PETDataset(h5_path, "test", idx)
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        preds, truths = [], []
        model = models[name]
        with torch.no_grad():
            for imgs, lbls in dl:
                preds.append(model(imgs).numpy())
                truths.append(lbls.numpy())
        preds  = np.concatenate(preds)
        truths = np.concatenate(truths)
        r, p = stats.pearsonr(preds, truths)
        print(f"{name}: r={r:.3f}, p={p:.3e}")


if __name__ == "__main__":
    for i in range(3):
        train_model(objective_idx=i)
    evaluate_all()