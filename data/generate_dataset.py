# data/generate_dataset.py
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
import h5py

IMG_SIZE = 64       # paper uses 128; 64 is fine for a course project
N_IMAGES = 10_000  # paper uses 50k; split 70/10/20

rng = np.random.default_rng(42)

def make_phantom(size=IMG_SIZE):
    """Simple elliptical organ phantom (liver-like)."""
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    # Outer body
    body = ((x/0.9)**2 + (y/0.85)**2) < 1
    # Organ (liver-like ellipse)
    organ = ((x/0.55)**2 + ((y-0.1)/0.45)**2) < 1
    img = np.zeros((size, size), dtype=np.float32)
    img[body] = 0.3
    img[organ] = 1.0
    return img, organ

def insert_lesion(img, organ_mask, contrast, radius_px, hot=True):
    """Insert a circular lesion; return updated image and lesion mask."""
    size = img.shape[0]
    ys, xs = np.where(organ_mask)
    margin = radius_px + 2
    # Sample centre strictly inside the organ
    for _ in range(200):
        idx = rng.integers(len(ys))
        cy, cx = ys[idx], xs[idx]
        if (cy > margin and cy < size-margin and
            cx > margin and cx < size-margin):
            break
    yg, xg = np.ogrid[:size, :size]
    mask = (yg-cy)**2 + (xg-cx)**2 <= radius_px**2
    mask &= organ_mask
    true_val = img[organ_mask].mean()
    lesion_val = true_val * contrast if hot else true_val / contrast
    img = img.copy()
    img[mask] = lesion_val
    return img, mask, (cy, cx)

def compute_labels(noisy, true_img, lesion_mask, bg_mask, fwhm_mm, pixel_size_mm=1.0):
    """
    Compute all three figures of merit (equations 4-6 in paper).
    Returns: inv_rmse, nsnr, inv_fwhm
    """
    # f_{1/RMSE}: inverse root mean square percent error over lesion pixels
    xi = noisy[lesion_mask]
    xi_true = true_img[lesion_mask]
    rmse = np.sqrt(np.mean(((xi - xi_true) / (xi_true + 1e-8))**2))
    inv_rmse = 1.0 / (rmse + 1e-6)

    # f_NSNR: normalised SNR (eq. 5)
    xh_bar = noisy[lesion_mask].mean()
    xb_bar = noisy[bg_mask].mean()
    xh_true = true_img[lesion_mask].mean()
    xb_true = true_img[bg_mask].mean()
    sigma_h = noisy[lesion_mask].std()
    sigma_b = noisy[bg_mask].std()
    sigma_bh = np.sqrt(0.5 * (sigma_b**2 + sigma_h**2)) + 1e-8
    nsnr = ((xh_bar - xb_bar) / ((xh_true - xb_true) + 1e-8)) / sigma_bh

    # f_{1/FWHM}: inverse FWHM of applied smoothing (eq. 6)
    inv_fwhm = 1.0 / (fwhm_mm + 1e-8)

    return np.float32(inv_rmse), np.float32(nsnr), np.float32(inv_fwhm)

def generate_dataset(n=N_IMAGES, save_path="data/pet_dataset.h5"):
    Path("data").mkdir(exist_ok=True)
    images, labels = [], []

    for i in range(n):
        # Step 1-2: phantom
        true_img, organ_mask = make_phantom()

        # Step 3: random zoom
        zf = rng.uniform(0.78, 1.22)
        true_img = zoom(true_img, zf, order=1)[:IMG_SIZE, :IMG_SIZE]
        pad = [(0, max(0, IMG_SIZE - true_img.shape[0])),
               (0, max(0, IMG_SIZE - true_img.shape[1]))]
        true_img = np.pad(true_img, pad)[:IMG_SIZE, :IMG_SIZE]
        organ_mask = true_img > 0.5

        # Step 4-6: hot lesion (diameter ~ N(11,25) px; contrast 1.5–8×)
        radius = int(np.clip(rng.normal(5.5, 2.5), 2, 10))
        contrast = rng.uniform(1.5, 8.0)
        try:
            true_img, lesion_mask, _ = insert_lesion(
                true_img, organ_mask, contrast, radius, hot=True)
        except Exception:
            continue

        # Step 7: background mask (same size, different location)
        bg_radius = radius
        bg_img, bg_mask, _ = insert_lesion(
            np.zeros_like(true_img), organ_mask, 1.0, bg_radius, hot=True)
        bg_mask = bg_mask & ~lesion_mask

        # Step 8: gamma noise (mimics reconstructed PET statistics)
        noise_scale = rng.uniform(0.05, 0.25)
        shape_param = 1.0 / (noise_scale**2 + 1e-8)
        noisy = rng.gamma(shape_param, true_img / (shape_param + 1e-8))
        noisy = noisy.astype(np.float32)

        # Step 9: Gaussian smoothing, W in [4,15] mm (= pixels here)
        fwhm_mm = rng.uniform(2.0, 8.0)
        sigma = fwhm_mm / (2.355)
        noisy = gaussian_filter(noisy, sigma=sigma)

        if lesion_mask.sum() < 5 or bg_mask.sum() < 5:
            continue

        inv_rmse, nsnr, inv_fwhm = compute_labels(
            noisy, true_img, lesion_mask, bg_mask, fwhm_mm)

        images.append(noisy[None])          # shape (1, H, W)
        labels.append([inv_rmse, nsnr, inv_fwhm])

        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{n} images generated")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Split: 70/10/20
    n_total = len(images)
    n_train = int(0.70 * n_total)
    n_val   = int(0.10 * n_total)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("train/images", data=images[:n_train])
        f.create_dataset("train/labels", data=labels[:n_train])
        f.create_dataset("val/images",   data=images[n_train:n_train+n_val])
        f.create_dataset("val/labels",   data=labels[n_train:n_train+n_val])
        f.create_dataset("test/images",  data=images[n_train+n_val:])
        f.create_dataset("test/labels",  data=labels[n_train+n_val:])

    print(f"Saved {n_total} images → {save_path}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_total-n_train-n_val}")

if __name__ == "__main__":
    generate_dataset()