# experiments/run_experiment.py
import numpy as np
import torch
from pathlib import Path

if __name__ == '__main__':
    from data.generate_dataset import generate_dataset, make_phantom, insert_lesion
    from models.cnn_objectives import train_model, load_models
    from reconstruction.pet_forward import make_system_matrix, simulate_sinogram
    from reconstruction.baselines import em_with_smoothing, map_reconstruction
    from optimizer.moeap import MOEAP
    from optimizer.r_moeap import RMOEAP
    from optimizer.kktpm import compute_kktpm, kktpm_summary
    from plots.visualize import plot_images, plot_pareto_front

    Path("results").mkdir(exist_ok=True)

    # ── Step 0: Generate dataset ──────────────────────────────────────────────
    print("=== Generating dataset ===")
    generate_dataset(n=10_000)

    # ── Step 1: Load or train CNN models ─────────────────────────────────────
    CHECKPOINT_DIR = "models/checkpoints"
    expected_files = [
        Path(CHECKPOINT_DIR) / "inv_rmse_best.pt",
        Path(CHECKPOINT_DIR) / "nsnr_best.pt",
        Path(CHECKPOINT_DIR) / "inv_fwhm_best.pt",
    ]
    all_present = all(f.exists() for f in expected_files)

    if all_present:
        print("\n=== Pretrained checkpoints found — loading models ===")
        for f in expected_files:
            print(f"  Found: {f}")
    else:
        missing = [str(f) for f in expected_files if not f.exists()]
        print("\n=== Some checkpoints missing — training from scratch ===")
        for m in missing:
            print(f"  Missing: {m}")
        for i in range(3):
            train_model(objective_idx=i, epochs=60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_models = load_models(save_dir=CHECKPOINT_DIR, device=device)
    print(f"  Models loaded on: {device}")

    # ── Step 2: Build PET forward model and simulate data ────────────────────
    print("\n=== Simulating PET data ===")
    IMG_SIZE = 64
    A = make_system_matrix(img_size=IMG_SIZE, n_angles=60)

    true_img, organ_mask = make_phantom(IMG_SIZE)
    true_img, lesion_mask, _ = insert_lesion(true_img, organ_mask,
                                              contrast=3.0, radius_px=5,
                                              hot=True)
    sinogram, scatter = simulate_sinogram(true_img, A, n_events=8e4)

    # ── Step 3: MOEAP with two objectives ────────────────────────────────────
    print("\n=== Running MOEAP (Poisson LL + NSNR) ===")
    moeap = MOEAP(sinogram, A, cnn_models, device,
                  objectives=["poisson_ll", "nsnr"],
                  pop_size=50, max_gen=100, img_size=IMG_SIZE)
    pop, obj_vals, fronts = moeap.run(verbose=True)
    np.save("results/moeap_obj.npy", obj_vals)
    np.save("results/moeap_pop.npy", np.array(pop))

    # ── Step 4: R-MOEAP with a reference point ───────────────────────────────
    print("\n=== Running R-MOEAP ===")
    ref_pt = obj_vals[fronts[0]].mean(axis=0)
    rmoeap = RMOEAP(sinogram, A, cnn_models, device,
                    objectives=["poisson_ll", "nsnr"],
                    reference_points=[ref_pt],
                    epsilon=0.05,
                    pop_size=50, max_gen=100, img_size=IMG_SIZE)
    r_pop, r_obj, r_fronts = rmoeap.run(verbose=True)
    np.save("results/rmoeap_obj.npy", r_obj)

    # ── Step 5: KKTPM convergence tracking ───────────────────────────────────
    print("\n=== Computing KKTPM ===")
    def eval_fn(img):
        return moeap._evaluate(img)

    # Sample small subset of front to keep runtime manageable
    
    front_idx = fronts[0]
    front_pop = [pop[i] for i in front_idx]
    front_obj = obj_vals[front_idx]
    kktpm_vals = compute_kktpm(front_obj, front_pop, eval_fn)
    kktpm_summary(kktpm_vals)

    # ── Step 6: Baselines ─────────────────────────────────────────────────────
    print("\n=== Running baselines ===")
    em_results  = em_with_smoothing(sinogram, A)
    map_results = map_reconstruction(sinogram, A)

    # ── Step 7: Plots ─────────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    plot_pareto_front(obj_vals, fronts,
                  em_results, map_results,
                  cnn_models, device, A, sinogram,
                  objectives=["Poisson LL", "NSNR"],
                  moeap_obj=moeap,                      # <-- add this
                  save_path="results/pareto_front.png")

    plot_images(pop, fronts, em_results, map_results,
                true_img, save_path="results/reconstruction_images.png")

    print("\n=== Done! Results saved to results/ ===")