# plots/visualize.py
import numpy as np
import matplotlib.pyplot as plt

def plot_pareto_front(obj_vals, fronts, em_results, map_results,
                      cnn_models, device, A, sinogram,
                      objectives, save_path="results/pareto_front.png"):
    from optimizer.moeap import MOEAP, poisson_ll
    from models.cnn_objectives import evaluate_cnn_objectives

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MOEAP Pareto front
    ax = axes[0]
    front0 = obj_vals[fronts[0]]
    ax.scatter(front0[:,0], front0[:,1], c='k', s=30, label='MOEAP', zorder=3)

    # EM + smoothing points
    em_x, em_y = [], []
    for fwhm, img in em_results:
        ll = poisson_ll(img, sinogram, A)
        nsnr_val = evaluate_cnn_objectives(img, cnn_models, device).get("nsnr", 0)
        em_x.append(ll); em_y.append(nsnr_val)
    ax.plot(em_x, em_y, 'b--o', markersize=5, label='EM+smooth')

    # MAP points
    map_x, map_y = [], []
    for beta, img in map_results:
        ll = poisson_ll(img, sinogram, A)
        nsnr_val = evaluate_cnn_objectives(img, cnn_models, device).get("nsnr", 0)
        map_x.append(ll); map_y.append(nsnr_val)
    ax.plot(map_x, map_y, 'r--s', markersize=5, label='MAP')

    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.legend()
    ax.set_title("Pareto front comparison")

    # Right: KKTPM per generation
    ax = axes[1]
    if hasattr(moeap_global, 'obj_history'):
        medians = [np.median(front) for front in moeap_global.obj_history]
        ax.plot(medians, 'k-')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Median KKTPM")
        ax.set_title("Convergence (KKTPM)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_images(pop, fronts, em_results, map_results,
                true_img, save_path="results/reconstruction_images.png"):
    """Show MOEAP compromise image vs EM vs MAP vs truth."""
    # Pick the median-crowding-distance solution from the front
    front_imgs = [pop[i] for i in fronts[0]]
    moeap_img = front_imgs[len(front_imgs)//2]

    # Best EM (middle of sweep)
    em_img = em_results[len(em_results)//2][1]

    # Mid MAP
    map_img = map_results[len(map_results)//2][1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(axes,
        [true_img, moeap_img, em_img, map_img],
        ["True", "MOEAP", "EM+smooth", "MAP"]):
        im = ax.imshow(img, cmap="hot", vmin=0)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")