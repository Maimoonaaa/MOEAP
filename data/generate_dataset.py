# data/generate_dataset.py
"""
Improved dataset generator for MOEAP.

Key upgrades over the baseline:
  1. Multi-organ phantoms (liver, lung, brain-like ellipses).
  2. Cold lesions in addition to hot lesions.
  3. Realistic PET noise: Poisson sinogram → reconstruction instead of
     direct gamma-noise on the image. This gives spatially correlated,
     reconstruction-accurate noise statistics.
  4. Background activity heterogeneity (random sub-regions at different uptake).
  5. Gaussian blur applied AFTER reconstruction — identical to clinical workflow.
  6. Compatible with OpenNeuro / TCIA preprocessed NIfTI slices when available
     (see load_real_slices() below).

HOW TO USE REAL DATA (optional upgrade):
  Download the TCIA "QIN HEADNECK" or "RIDER Lung PET-CT" collections via
  the TCIA REST API, convert DICOM → NIfTI with dcm2niix, then call
      generate_dataset(real_nifti_dir="path/to/nifti_slices/")
  The loader will extract 2-D axial slices, normalise them, and use them as
  "true images" instead of synthetic phantoms.

Dependencies: numpy, scipy, h5py, (optional) nibabel, scikit-image
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.sparse import csr_matrix
from pathlib import Path
import h5py

IMG_SIZE  = 128
N_IMAGES  = 10_000
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Phantom generation
# ─────────────────────────────────────────────────────────────────────────────

def _ellipse_mask(size, cx, cy, rx, ry, angle_deg=0.0):
    """Return boolean mask for a rotated ellipse."""
    y, x = np.ogrid[:size, :size]
    xc = x - cx; yc = y - cy
    a  = np.deg2rad(angle_deg)
    xr =  xc*np.cos(a) + yc*np.sin(a)
    yr = -xc*np.sin(a) + yc*np.cos(a)
    return (xr/rx)**2 + (yr/ry)**2 < 1.0


def make_phantom(size=IMG_SIZE, organ_type=None):
    """
    Generate a synthetic organ phantom with heterogeneous background.

    organ_type: 'liver' | 'lung' | 'brain' | None (random)
    Returns: (image float32, organ_mask bool)
    """
    if organ_type is None:
        organ_type = rng.choice(['liver', 'lung', 'brain'])

    img = np.zeros((size, size), dtype=np.float32)
    cx  = size // 2; cy = size // 2

    # ── body outline ──────────────────────────────────────────────────────────
    body = _ellipse_mask(size, cx, cy, size*0.44, size*0.42)
    img[body] = rng.uniform(0.15, 0.35)          # low body background

    # ── primary organ ─────────────────────────────────────────────────────────
    if organ_type == 'liver':
        organ = _ellipse_mask(size, cx + size*0.05, cy - size*0.05,
                              size*0.28, size*0.22, angle_deg=rng.uniform(-15, 15))
        uptake = rng.uniform(0.8, 1.4)

    elif organ_type == 'lung':
        # Two lung lobes
        lobe_l = _ellipse_mask(size, cx - size*0.18, cy, size*0.14, size*0.20)
        lobe_r = _ellipse_mask(size, cx + size*0.18, cy, size*0.14, size*0.20)
        organ  = lobe_l | lobe_r
        uptake = rng.uniform(0.4, 0.8)

    else:  # brain
        organ = _ellipse_mask(size, cx, cy, size*0.36, size*0.30)
        grey  = _ellipse_mask(size, cx, cy, size*0.36, size*0.30) & \
               ~_ellipse_mask(size, cx, cy, size*0.26, size*0.22)
        img[grey] = rng.uniform(0.6, 1.0)        # grey matter ring
        organ  = _ellipse_mask(size, cx, cy, size*0.36, size*0.30)
        uptake = rng.uniform(1.0, 1.6)

    img[organ] = uptake

    # ── background heterogeneity: 3-5 random sub-regions ─────────────────────
    n_bg = rng.integers(3, 6)
    for _ in range(n_bg):
        bx = rng.integers(5, size - 5)
        by = rng.integers(5, size - 5)
        br = rng.integers(3, 10)
        bv = rng.uniform(0.1, 0.5)
        bg_mask = _ellipse_mask(size, bx, by, br, br)
        bg_mask &= body & ~organ
        img[bg_mask] = bv

    # ── smooth slightly to remove sharp edges ─────────────────────────────────
    img = gaussian_filter(img, sigma=0.8)
    img = np.clip(img, 0, None)
    return img.astype(np.float32), organ


def insert_lesion(img, organ_mask, contrast, radius_px, hot=True):
    """Insert a circular lesion inside the organ. Returns (img, mask, centre)."""
    size = img.shape[0]
    ys, xs = np.where(organ_mask)
    margin  = radius_px + 3
    for _ in range(500):
        idx = rng.integers(len(ys))
        cy, cx = ys[idx], xs[idx]
        if margin < cy < size - margin and margin < cx < size - margin:
            break
    yg, xg   = np.ogrid[:size, :size]
    mask      = (yg - cy)**2 + (xg - cx)**2 <= radius_px**2
    mask     &= organ_mask
    true_val  = img[organ_mask].mean()
    lesion_val = true_val * contrast if hot else true_val / contrast
    img = img.copy()
    img[mask] = lesion_val
    return img, mask, (cy, cx)


# ─────────────────────────────────────────────────────────────────────────────
# Realistic PET forward / reconstruction noise
# ─────────────────────────────────────────────────────────────────────────────

def _make_system_matrix(img_size=IMG_SIZE, n_angles=30):
    """Lightweight 2-D parallel-beam system matrix."""
    n_det  = img_size
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    cx = cy = img_size / 2.0
    rows, cols, data = [], [], []
    row_id = 0
    for angle in angles:
        ca, sa = np.cos(angle), np.sin(angle)
        for d in range(n_det):
            offset = d - n_det / 2.0 + 0.5
            t_vals = np.linspace(-img_size / 2, img_size / 2, img_size * 2)
            xs = cx + t_vals * ca - offset * sa
            ys = cy + t_vals * sa + offset * ca
            xi = np.floor(xs).astype(int); yi = np.floor(ys).astype(int)
            valid = (xi >= 0) & (xi < img_size) & (yi >= 0) & (yi < img_size)
            for pid in np.unique(xi[valid] + yi[valid] * img_size):
                rows.append(row_id); cols.append(pid); data.append(1.0)
            row_id += 1
    A = csr_matrix((data, (rows, cols)), shape=(row_id, img_size * img_size))
    rs = np.array(A.sum(axis=1)).flatten(); rs[rs == 0] = 1
    return A.multiply(1.0 / rs[:, None])


# Build once at module import — dense (180 angles for high fidelity)
_A_SMALL = _make_system_matrix(IMG_SIZE, n_angles=180)


def pet_noise_reconstruction(true_img, n_events=5e4, scatter_frac=0.1):
    """
    Simulate PET acquisition + 10-iter MLEM reconstruction.
    Gives spatially-correlated noise that matches real PET statistics,
    far more realistic than direct gamma noise on pixels.
    """
    A  = _A_SMALL
    yf = A @ true_img.flatten()
    yf = yf / (yf.sum() + 1e-10) * n_events
    sc = scatter_frac * yf.mean() * np.ones_like(yf)
    y  = np.random.poisson(yf + sc).astype(float)

    # 10-iter MLEM
    x   = np.ones(A.shape[1])
    At1 = A.T @ np.ones(A.shape[0])
    for _ in range(10):
        yb  = np.clip(A @ x + sc, 1e-10, None)
        x   = x * (A.T @ (y / yb)) / (At1 + 1e-10)
        x   = np.clip(x, 0, None)
    return x.reshape(true_img.shape).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: load real NIfTI slices
# ─────────────────────────────────────────────────────────────────────────────

def load_real_slices(nifti_dir, target_size=IMG_SIZE, max_slices=2000):
    """
    Load 2-D axial slices from NIfTI files (requires nibabel).
    Use with TCIA / OpenNeuro data for non-synthetic phantoms.

    Steps to get data:
      1. Download via TCIA Downloader: https://www.cancerimagingarchive.net
         Recommended collections:
           - "QIN HEADNECK" (PET/CT, free access)
           - "RIDER Lung PET-CT"
           - "ACRIN-FLT-Breast"
         All freely available after TCIA account registration.
      2. Convert DICOM → NIfTI:
           pip install dcm2niix
           dcm2niix -o ./nifti_out -z y ./dicom_folder
      3. Call this function with nifti_dir="./nifti_out"

    Returns list of (H, W) float32 arrays normalised to [0, 1].
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    slices = []
    for f in sorted(Path(nifti_dir).rglob("*.nii*")):
        vol = nib.load(str(f)).get_fdata().astype(np.float32)
        vol = np.clip(vol, 0, None)
        vmax = vol.max()
        if vmax < 1e-6:
            continue
        vol /= vmax
        # Take every other axial slice from the middle 60% of the volume
        nz = vol.shape[2]
        z0, z1 = int(0.2 * nz), int(0.8 * nz)
        for z in range(z0, z1, 2):
            sl = vol[:, :, z]
            # Resize to target_size × target_size
            zf = target_size / sl.shape[0]
            sl = zoom(sl, zf, order=1)[:target_size, :target_size]
            pad = [(0, max(0, target_size - sl.shape[0])),
                   (0, max(0, target_size - sl.shape[1]))]
            sl = np.pad(sl, pad)[:target_size, :target_size]
            if sl.max() > 0.05:          # skip near-empty slices
                slices.append(sl.astype(np.float32))
            if len(slices) >= max_slices:
                return slices
    print(f"Loaded {len(slices)} real NIfTI slices from {nifti_dir}")
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Label computation (paper Eqs. 4-6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_labels(noisy, true_img, lesion_mask, bg_mask, fwhm_px):
    """
    Returns (inv_rmse, nsnr, inv_fwhm) — all higher = better.
    """
    # f_{1/RMSE}
    xi      = noisy[lesion_mask]
    xi_true = true_img[lesion_mask]
    rmse    = np.sqrt(np.mean(((xi - xi_true) / (xi_true + 1e-8))**2))
    inv_rmse = 1.0 / (rmse + 1e-6)

    # f_NSNR
    xh_bar = noisy[lesion_mask].mean()
    xb_bar = noisy[bg_mask].mean()
    xh_t   = true_img[lesion_mask].mean()
    xb_t   = true_img[bg_mask].mean()
    sh     = noisy[lesion_mask].std()
    sb     = noisy[bg_mask].std()
    sigma_bh = np.sqrt(0.5 * (sb**2 + sh**2)) + 1e-8
    nsnr   = ((xh_bar - xb_bar) / ((xh_t - xb_t) + 1e-8)) / sigma_bh

    # f_{1/FWHM}
    inv_fwhm = 1.0 / (fwhm_px + 1e-8)

    return np.float32(inv_rmse), np.float32(nsnr), np.float32(inv_fwhm)


# ─────────────────────────────────────────────────────────────────────────────
# Main dataset generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n=N_IMAGES,
                     save_path="data/pet_dataset.h5",
                     real_nifti_dir=None,
                     use_pet_noise=True):
    """
    Generate training dataset.

    Args:
        n              : number of images to generate
        save_path      : output HDF5 path
        real_nifti_dir : if given, use real NIfTI slices as base phantoms
        use_pet_noise  : if True, use PET forward→reconstruction noise
                         (more realistic; slower); else use gamma noise
    """
    Path("data").mkdir(exist_ok=True)

    real_slices = []
    if real_nifti_dir is not None:
        real_slices = load_real_slices(real_nifti_dir)
        print(f"Using {len(real_slices)} real slices as base phantoms")

    images, labels = [], []
    organ_types = ['liver', 'lung', 'brain']

    for i in range(n):
        # ── Step 1: get base phantom ──────────────────────────────────────────
        if real_slices:
            idx      = rng.integers(len(real_slices))
            true_img = real_slices[idx].copy()
            # Create organ mask from intensity thresholding
            organ_mask = true_img > (true_img.mean() * 0.6)
        else:
            otype    = organ_types[i % len(organ_types)]
            true_img, organ_mask = make_phantom(IMG_SIZE, organ_type=otype)

        # ── Step 2: random zoom ───────────────────────────────────────────────
        zf       = rng.uniform(0.80, 1.20)
        true_img = zoom(true_img, zf, order=1)[:IMG_SIZE, :IMG_SIZE]
        pad      = [(0, max(0, IMG_SIZE - true_img.shape[0])),
                    (0, max(0, IMG_SIZE - true_img.shape[1]))]
        true_img = np.pad(true_img, pad)[:IMG_SIZE, :IMG_SIZE]
        organ_mask = true_img > true_img.mean() * 0.5

        # ── Step 3: insert lesion (50% hot, 50% cold) ─────────────────────────
        radius   = int(np.clip(rng.normal(5.5, 2.0), 2, 9))
        contrast = rng.uniform(1.5, 8.0)
        hot      = rng.random() > 0.5
        try:
            true_img, lesion_mask, _ = insert_lesion(
                true_img, organ_mask, contrast, radius, hot=hot)
        except Exception:
            continue

        # ── Step 4: background mask ───────────────────────────────────────────
        bg_radius = radius
        try:
            _, bg_mask, _ = insert_lesion(
                np.zeros_like(true_img), organ_mask, 1.0, bg_radius, hot=True)
            bg_mask = bg_mask & ~lesion_mask
        except Exception:
            continue

        if lesion_mask.sum() < 5 or bg_mask.sum() < 5:
            continue

        # ── Step 5: noise ─────────────────────────────────────────────────────
        if use_pet_noise:
            # Scaled up n_events to heavily reduce Poisson shot noise for 128x128
            n_events = rng.choice([5e5, 1e6, 2e6, 5e6])
            noisy    = pet_noise_reconstruction(true_img, n_events=n_events)
        else:
            # Fallback: gamma noise (fast but less realistic)
            noise_scale = rng.uniform(0.05, 0.25)
            sp          = 1.0 / (noise_scale**2 + 1e-8)
            noisy       = rng.gamma(sp, true_img / (sp + 1e-8)).astype(np.float32)

        # ── Step 6: post-reconstruction Gaussian blur ─────────────────────────
        fwhm_px = rng.uniform(2.0, 8.0)
        sigma   = fwhm_px / 2.355
        noisy   = gaussian_filter(noisy, sigma=sigma).astype(np.float32)

        # ── Step 7: labels ────────────────────────────────────────────────────
        inv_rmse, nsnr, inv_fwhm = compute_labels(
            noisy, true_img, lesion_mask, bg_mask, fwhm_px)

        # Skip degenerate samples
        if not (np.isfinite(inv_rmse) and np.isfinite(nsnr) and np.isfinite(inv_fwhm)):
            continue

        images.append(noisy[None])               # (1, H, W)
        labels.append([inv_rmse, nsnr, inv_fwhm])

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n} images generated")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    n_tot  = len(images)
    n_tr   = int(0.70 * n_tot)
    n_val  = int(0.10 * n_tot)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("train/images", data=images[:n_tr])
        f.create_dataset("train/labels", data=labels[:n_tr])
        f.create_dataset("val/images",   data=images[n_tr:n_tr+n_val])
        f.create_dataset("val/labels",   data=labels[n_tr:n_tr+n_val])
        f.create_dataset("test/images",  data=images[n_tr+n_val:])
        f.create_dataset("test/labels",  data=labels[n_tr+n_val:])

    print(f"\nSaved {n_tot} images → {save_path}")
    print(f"  Train: {n_tr} | Val: {n_val} | Test: {n_tot-n_tr-n_val}")
    return save_path


if __name__ == "__main__":
    generate_dataset(n=10_000, use_pet_noise=True)