import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter


def plot_pareto_front(obj_vals, fronts, em_results, map_results,
                      cnn_models, device, A, sinogram,
                      r_obj=None, r_fronts=None,
                      objectives=("Poisson LL", "NSNR"),
                      kktpm_history=None,
                      save_path="results/pareto_front.png"):

    from optimizer.moeap import poisson_ll, evaluate_cnn_objectives

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    front0 = obj_vals[fronts[0]]
    ax1.scatter(front0[:,0], front0[:,1], c='k', s=40,
                label='MOEAP front', zorder=4)
    ax1.scatter(obj_vals[:,0], obj_vals[:,1], c='gray', s=10,
                alpha=0.3, label='MOEAP all', zorder=3)

    if r_obj is not None and r_fronts is not None:
        rf0 = r_obj[r_fronts[0]]
        ax1.scatter(rf0[:,0], rf0[:,1], c='green', s=40, marker='^',
                    label='R-MOEAP front', zorder=4)

    em_x, em_y = [], []
    for fwhm, img in em_results:
        ll  = poisson_ll(img, sinogram, A)
        nsnr = evaluate_cnn_objectives(img, cnn_models, device).get("nsnr", 0)
        em_x.append(ll); em_y.append(nsnr)
    ax1.plot(em_x, em_y, 'b--o', markersize=5, label='EM+smooth')

    map_x, map_y = [], []
    for beta, img in map_results:
        ll  = poisson_ll(img, sinogram, A)
        nsnr = evaluate_cnn_objectives(img, cnn_models, device).get("nsnr", 0)
        map_x.append(ll); map_y.append(nsnr)
    ax1.plot(map_x, map_y, 'r--s', markersize=5, label='MAP')

    ax1.set_xlabel(objectives[0]); ax1.set_ylabel(objectives[1])
    ax1.set_title("Pareto front comparison")
    ax1.legend(fontsize=7)


    ax2 = fig.add_subplot(gs[0, 1])
    if kktpm_history and len(kktpm_history) > 0:
        ax2.plot(kktpm_history, 'k-', linewidth=1.5)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Median KKTPM")
        ax2.set_title("Convergence — KKTPM should decrease")
        ax2.axhline(0, color='r', linestyle='--', alpha=0.4, label='Optimal=0')
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "No KKTPM history\n(enable per-gen tracking)",
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Convergence (KKTPM)")

    ax3 = fig.add_subplot(gs[0, 2])
    if hasattr(obj_vals, '__len__'):
        for m, name in enumerate(objectives):
            ax3.hist(obj_vals[:, m], bins=20, alpha=0.5, label=name)
        ax3.set_xlabel("Objective value")
        ax3.set_ylabel("Count")
        ax3.set_title("Final population objective distribution")
        ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[1, 0])
    fwhm_vals = [r[0] for r in em_results]
    ax4.plot(fwhm_vals, em_x, 'b-o', markersize=4, label='Poisson LL')
    ax4r = ax4.twinx()
    ax4r.plot(fwhm_vals, em_y, 'g--s', markersize=4, label='NSNR')
    ax4.set_xlabel("Post-smoothing FWHM (px)")
    ax4.set_ylabel("Poisson LL", color='b')
    ax4r.set_ylabel("NSNR", color='g')
    ax4.set_title("EM: objective vs smoothing strength")

    ax5 = fig.add_subplot(gs[1, 1])
    beta_vals = [r[0] for r in map_results]
    ax5.semilogx(beta_vals, map_x, 'r-o', markersize=4, label='Poisson LL')
    ax5r = ax5.twinx()
    ax5r.semilogx(beta_vals, map_y, 'm--s', markersize=4, label='NSNR')
    ax5.set_xlabel("MAP penalty β (log scale)")
    ax5.set_ylabel("Poisson LL", color='r')
    ax5r.set_ylabel("NSNR", color='m')
    ax5.set_title("MAP: objective vs penalty strength")

    ax6 = fig.add_subplot(gs[1, 2])
    if len(front0) > 2:
        from scipy.spatial.distance import pdist
        dists = pdist(front0, metric='euclidean')
        ax6.hist(dists, bins=15, color='steelblue', edgecolor='k', alpha=0.7)
        ax6.axvline(dists.mean(), color='r', linestyle='--',
                    label=f'Mean={dists.mean():.1f}')
        ax6.set_xlabel("Pairwise Euclidean distance")
        ax6.set_ylabel("Count")
        ax6.set_title("Pareto front diversity")
        ax6.legend(fontsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_images(pop, fronts, em_results, map_results,
                true_img, sinogram=None, A=None,
                save_path="results/reconstruction_images.png"):
    
    fi = fronts[0]
    n_front = len(fi)
    idx_low  = fi[0]                  # one end of front
    idx_mid  = fi[n_front // 2]       # compromise
    idx_high = fi[-1]                 # other end

    moeap_low  = pop[idx_low]
    moeap_mid  = pop[idx_mid]
    moeap_high = pop[idx_high]

    em_img  = em_results[len(em_results)  // 2][1]
    map_img = map_results[len(map_results) // 2][1]

    imgs   = [true_img, moeap_low, moeap_mid, moeap_high, em_img, map_img]
    titles = ["True", "MOEAP\n(end A)", "MOEAP\n(compromise)",
              "MOEAP\n(end B)", "EM+smooth", "MAP"]


    disp_max = true_img.max() * 3
    imgs_disp = [np.clip(im, 0, disp_max) for im in imgs]

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.35)

    vmax = true_img.max() * 2
    for j, (img, title) in enumerate(zip(imgs_disp, titles)):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(img, cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j, (img, title) in enumerate(zip(imgs_disp, titles)):
        ax = fig.add_subplot(gs[1, j])
        err = np.abs(img - true_img) / (true_img + 1e-8)
        err = np.clip(err, 0, 2)
        im  = ax.imshow(err, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f"% Error\n{title.split(chr(10))[0]}", fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_prof = fig.add_subplot(gs[2, :3])
    mid_row = true_img.shape[0] // 2
    colors  = ['k', 'purple', 'blue', 'cyan', 'green', 'red']
    for img, title, color in zip(imgs_disp, titles, colors):
        profile = img[mid_row, :]
        label   = title.replace('\n', ' ')
        ax_prof.plot(profile, color=color, label=label,
                     linewidth=1.5 if 'True' in label else 1.0,
                     linestyle='-' if 'True' in label else '--')
    ax_prof.set_xlabel("Pixel position (horizontal)")
    ax_prof.set_ylabel("Intensity")
    ax_prof.set_title("Horizontal profile through image centre")
    ax_prof.legend(fontsize=7, ncol=3)

    ax_bar = fig.add_subplot(gs[2, 3:])
    rmse_vals = []
    for img in imgs_disp:
        rmse = np.sqrt(np.mean(((img - true_img) /
                                (true_img + 1e-8))**2))
        rmse_vals.append(rmse)
    bar_colors = ['k', 'purple', 'blue', 'cyan', 'green', 'red']
    bars = ax_bar.bar([t.replace('\n', ' ') for t in titles],
                      rmse_vals, color=bar_colors, alpha=0.75,
                      edgecolor='black')
    ax_bar.set_ylabel("Normalised RMSE (lower=better)")
    ax_bar.set_title("Quantitation error per method")
    ax_bar.tick_params(axis='x', labelsize=7)
    for bar, val in zip(bars, rmse_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")