# experiments/run_experiment.py
"""
Updated experiment runner.
Wires together: improved dataset, ResNet CNNs, per-gen KKTPM,
hypervolume tracking, and the richer visualisation dashboard.
"""

import numpy as np
import torch
from pathlib import Path

if __name__ == '__main__':
    from data.generate_dataset       import generate_dataset, make_phantom, insert_lesion, load_real_slices
    from models.cnn_objectives        import train_model, load_models
    from reconstruction.pet_forward   import make_system_matrix, simulate_sinogram
    from reconstruction.baselines     import em_with_smoothing, map_reconstruction
    from optimizer.moeap              import MOEAP
    from optimizer.r_moeap            import RMOEAP
    from optimizer.kktpm              import compute_kktpm, kktpm_summary
    from plots.visualize              import (plot_pareto_front, plot_images,
                                              plot_training_curves)

    Path("results").mkdir(exist_ok=True)

    # --- Configuration Toggle ---
    USE_REAL_DATA = False        # Toggle between Synthetic and Real datasets
    IGNORE_FWHM = False
    FORCE_SKIP_TRAINING = False  # Set to True to completely skip training

    if USE_REAL_DATA:
        H5_PATH = "data/pet_dataset_real.h5"
        CKPT = "models/checkpoints_real"
        REAL_NIFTI_DIR = "data/nifti_slices"
    else:
        H5_PATH = "data/pet_dataset_synthetic.h5"
        CKPT = "models/checkpoints_synthetic"
        REAL_NIFTI_DIR = None

    # ── Step 0: dataset ───────────────────────────────────────────────────────
    h5_path = Path(H5_PATH)
    if h5_path.exists():
        print(f"=== Dataset {h5_path} already exists — skipping generation ===")
    else:
        print("=== Generating dataset ===")
        generate_dataset(n=10_000, save_path=H5_PATH, real_nifti_dir=REAL_NIFTI_DIR)

    # ── Step 1: CNN models ────────────────────────────────────────────────────
    train_objs = ["inv_rmse", "nsnr"] if IGNORE_FWHM else ["inv_rmse", "nsnr", "inv_fwhm"]
    needed = [Path(CKPT) / f"{n}_best.pt" for n in train_objs]
    
    if FORCE_SKIP_TRAINING or all(f.exists() for f in needed):
        print("\n=== Skipping training phase ===")
    else:
        print(f"\n=== Training ResNet CNN objective models in {CKPT} ===")
        for i, obj in enumerate(["inv_rmse", "nsnr", "inv_fwhm"]):
            if obj in train_objs:
                train_model(h5_path=H5_PATH, objective_idx=i, epochs=80, save_dir=CKPT)

    # Plot training curves if history files exist
    plot_training_curves(save_dir=CKPT, save_path="results/training_curves.png")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_models = load_models(save_dir=CKPT, device=device, ignore_fwhm=IGNORE_FWHM)
    print(f"  Models loaded on {device}")

    # ── Step 2: PET forward model and data ────────────────────────────────────
    print("\n=== Simulating PET data ===")
    IMG_SIZE = 128
    A = make_system_matrix(img_size=IMG_SIZE, n_angles=180)

    if USE_REAL_DATA:
        real_slices = load_real_slices(REAL_NIFTI_DIR, target_size=IMG_SIZE)
        # Pick a median slice to ensure it's solidly inside the body cavity
        true_img = real_slices[len(real_slices) // 2]
        organ_mask = true_img > (true_img.mean() * 0.6)
    else:
        true_img, organ_mask = make_phantom(IMG_SIZE)
    
    true_img, lesion_mask, _ = insert_lesion(true_img, organ_mask,
                                              contrast=3.0, radius_px=5, hot=True)
    sinogram, scatter = simulate_sinogram(true_img, A, n_events=5e6)

    # ── Step 3: MOEAP ─────────────────────────────────────────────────────────
    print("\n=== Running MOEAP ===")
    moeap = MOEAP(sinogram, A, cnn_models, device,
                  objectives=["poisson_ll", "nsnr"],
                  pop_size=50, max_gen=100, img_size=IMG_SIZE,
                  kktpm_every=10, kktpm_front_size=5)

    pop, obj_vals, fronts = moeap.run(verbose=True)
    np.save("results/moeap_obj.npy",  obj_vals)
    np.save("results/moeap_pop.npy",  np.array(pop))
    np.save("results/kktpm_hist.npy", np.array(moeap.kktpm_history))
    np.save("results/hv_hist.npy",    np.array(moeap.hv_history))

    # ── Step 4: R-MOEAP ───────────────────────────────────────────────────────
    print("\n=== Running R-MOEAP ===")
    ref_pt = obj_vals[fronts[0]].mean(axis=0)
    rmoeap = RMOEAP(sinogram, A, cnn_models, device,
                    objectives=["poisson_ll", "nsnr"],
                    reference_points=[ref_pt], epsilon=0.05,
                    pop_size=50, max_gen=100, img_size=IMG_SIZE)
    r_pop, r_obj, r_fronts = rmoeap.run(verbose=True)
    np.save("results/rmoeap_obj.npy", r_obj)

    # ── Step 5: KKTPM on final front ──────────────────────────────────────────
    print("\n=== Computing final KKTPM ===")
    def eval_fn(img):
        return moeap._evaluate(img)

    front_idx = fronts[0]
    kktpm_vals, lambdas = compute_kktpm(
        obj_vals[front_idx],
        [pop[i] for i in front_idx],
        eval_fn, h=1e-3, n_sample=80)
    kktpm_summary(kktpm_vals)
    np.save("results/kktpm_final.npy",  kktpm_vals)
    np.save("results/lambdas_final.npy", lambdas)

    print("\n=== Running baselines ===")
    em_results  = em_with_smoothing(sinogram, A)
    map_results = map_reconstruction(sinogram, A)

    print("\n=== Generating plots ===")
    plot_pareto_front(
        obj_vals, fronts, em_results, map_results,
        cnn_models, device, A, sinogram,
        r_obj=r_obj, r_fronts=r_fronts,
        objectives=["Poisson LL", "NSNR"],
        kktpm_history=moeap.kktpm_history,
        kktpm_full_history=moeap.kktpm_full_history,
        hv_history=moeap.hv_history,
        obj_history=moeap.obj_history,
        save_path="results/pareto_front.png"
    )

    plot_images(pop, fronts, em_results, map_results, true_img,
                sinogram=sinogram, A=A,
                save_path="results/reconstruction_images.png")

    print("\n=== Done! Results saved to results/ ===")