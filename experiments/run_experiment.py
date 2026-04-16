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
    from data.generate_dataset       import generate_dataset, make_phantom, insert_lesion
    from models.cnn_objectives        import train_model, load_models
    from reconstruction.pet_forward   import make_system_matrix, simulate_sinogram
    from reconstruction.baselines     import em_with_smoothing, map_reconstruction
    from optimizer.moeap              import MOEAP
    from optimizer.r_moeap            import RMOEAP
    from optimizer.kktpm              import compute_kktpm, kktpm_summary
    from plots.visualize              import (plot_pareto_front, plot_images,
                                              plot_training_curves)

    Path("results").mkdir(exist_ok=True)

    # ── Step 0: dataset ───────────────────────────────────────────────────────
    print("=== Generating dataset ===")
    # generate_dataset(n=3_000, use_pet_noise=True)
    # To use real TCIA NIfTI slices instead, run:
    generate_dataset(n=10_000, real_nifti_dir="data/nifti_slices/")

    # ── Step 1: CNN models ────────────────────────────────────────────────────
    CKPT = "models/checkpoints"
    needed = [Path(CKPT) / f"{n}_best.pt"
              for n in ["inv_rmse", "nsnr", "inv_fwhm"]]
    if all(f.exists() for f in needed):
        print("\n=== Checkpoints found — skipping training ===")
    else:
        print("\n=== Training ResNet CNN objective models ===")
        for i in range(3):
            train_model(objective_idx=i, epochs=80)

    # Plot training curves if history files exist
    plot_training_curves(save_dir=CKPT, save_path="results/training_curves.png")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_models = load_models(save_dir=CKPT, device=device)
    print(f"  Models loaded on {device}")

    # ── Step 2: PET forward model and data ────────────────────────────────────
    print("\n=== Simulating PET data ===")
    IMG_SIZE = 64
    A = make_system_matrix(img_size=IMG_SIZE, n_angles=60)

    true_img, organ_mask = make_phantom(IMG_SIZE)
    true_img, lesion_mask, _ = insert_lesion(true_img, organ_mask,
                                              contrast=3.0, radius_px=5, hot=True)
    sinogram, scatter = simulate_sinogram(true_img, A, n_events=8e4)

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