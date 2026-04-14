# experiments/run_experiment.py
"""
Full pipeline: train CNNs → simulate PET → run MOEAP → compare baselines.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.generate_dataset import generate_dataset
from models.cnn_objectives import train_model, load_models, OBJECTIVE_NAMES
from reconstruction.pet_forward import make_system_matrix, simulate_sinogram
from reconstruction.baselines import em_with_smoothing, map_reconstruction
from optimizer.moeap import MOEAP
from optimizer.r_moeap import RMOEAP
from optimizer.kktpm import compute_kktpm, kktpm_summary
from plots.visualize import plot_pareto_front, plot_images

Path("results").mkdir(exist_ok=True)

# ── Step 0: Generate dataset ─────────────────────────────────────────────────
print("=== Generating dataset ===")
generate_dataset(n=10_000)

# ── Step 1: Train CNN objective models ───────────────────────────────────────
print("\n=== Training CNN models ===")
for i in range(3):
    train_model(objective_idx=i, epochs=60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_models = load_models(device=device)

# ── Step 2: Build PET forward model and simulate data ────────────────────────
print("\n=== Simulating PET data ===")
IMG_SIZE = 64
A = make_system_matrix(img_size=IMG_SIZE, n_angles=60)

# Simple elliptical phantom as true image
from data.generate_dataset import make_phantom, insert_lesion
true_img, organ_mask = make_phantom(IMG_SIZE)
true_img, lesion_mask, _ = insert_lesion(true_img, organ_mask, contrast=3.0,
                                          radius_px=5, hot=True)
sinogram, scatter = simulate_sinogram(true_img, A, n_events=8e4)

# ── Step 3: MOEAP with two objectives ────────────────────────────────────────
print("\n=== Running MOEAP (Poisson LL + NSNR) ===")
moeap = MOEAP(sinogram, A, cnn_models, device,
              objectives=["poisson_ll", "nsnr"],
              pop_size=50, max_gen=100, img_size=IMG_SIZE)
pop, obj_vals, fronts = moeap.run(verbose=True)
np.save("results/moeap_obj.npy", obj_vals)
np.save("results/moeap_pop.npy", np.array(pop))

# ── Step 4: R-MOEAP with a reference point ───────────────────────────────────
print("\n=== Running R-MOEAP ===")
# Choose a reference point in objective space near the MOEAP front
ref_pt = obj_vals[fronts[0]].mean(axis=0)
rmoeap = RMOEAP(sinogram, A, cnn_models, device,
                objectives=["poisson_ll", "nsnr"],
                reference_points=[ref_pt],
                epsilon=0.05,
                pop_size=50, max_gen=100, img_size=IMG_SIZE)
r_pop, r_obj, r_fronts = rmoeap.run(verbose=True)
np.save("results/rmoeap_obj.npy", r_obj)

# ── Step 5: KKTPM convergence tracking ───────────────────────────────────────
print("\n=== Computing KKTPM ===")
def eval_fn(img): return moeap._evaluate(img)

front_pop = [pop[i] for i in fronts[0]]
front_obj = obj_vals[fronts[0]]
kktpm_vals = compute_kktpm(front_obj, front_pop, eval_fn)
kktpm_summary(kktpm_vals)

# ── Step 6: Baselines ─────────────────────────────────────────────────────────
print("\n=== Running baselines ===")
em_results  = em_with_smoothing(sinogram, A)
map_results = map_reconstruction(sinogram, A)

# ── Step 7: Plots ─────────────────────────────────────────────────────────────
print("\n=== Generating plots ===")
plot_pareto_front(obj_vals, fronts,
                  em_results, map_results,
                  cnn_models, device, A, sinogram,
                  objectives=["Poisson LL", "NSNR"],
                  save_path="results/pareto_front.png")

plot_images(pop, fronts, em_results, map_results,
            true_img, save_path="results/reconstruction_images.png")