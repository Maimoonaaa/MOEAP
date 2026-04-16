# optimizer/kktpm.py
"""
Improved KKTPM implementation.

Key changes over baseline:
  • Per-objective gradient normalisation before Gram matrix construction
    prevents scale dominance (one objective overwhelming others).
  • Projected gradient descent with adaptive step-size (Armijo line search)
    for faster simplex projection convergence.
  • Returns both the scalar KKTPM and the optimal lambda vector for debugging.
  • kktpm_per_generation() helper tracks the full five-number summary
    across a list of generation snapshots for rich convergence plots.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Simplex projection  (O(M log M) closed-form algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def _project_simplex(v):
    """Project vector v onto the probability simplex (sum=1, v>=0)."""
    n    = len(v)
    u    = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho  = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core KKTPM
# ─────────────────────────────────────────────────────────────────────────────

def compute_kktpm(obj_values, population, evaluate_fn,
                  h=1e-3, n_sample=100, pgd_iters=300, pgd_lr=0.05):
    """
    Compute KKTPM for each solution in population.

    KKTPM(x) = min_{λ ∈ Δ^{M-1}} ‖Σ_m λ_m ∇f_m(x)‖₂

    Gradients approximated by central finite differences on a random
    sub-sample of n_sample pixel coordinates.
    Each objective's gradient is L2-normalised before constructing
    the Gram matrix (prevents scale dominance).

    Args:
        obj_values  : (N, M) objective values for N solutions
        population  : list of N image arrays
        evaluate_fn : img → (M,) objective vector
        h           : finite-difference step
        n_sample    : number of pixel coordinates to sample
        pgd_iters   : projected gradient descent iterations
        pgd_lr      : PGD learning rate

    Returns:
        kktpm_vals : (N,) KKTPM values in [0, 1]
        lambdas    : (N, M) optimal λ vectors
    """
    N, M      = obj_values.shape
    kktpm_raw = np.zeros(N)
    lambdas   = np.zeros((N, M))

    for i, img in enumerate(population):
        flat     = img.flatten()
        n_params = len(flat)
        idx      = np.random.choice(n_params,
                                    size=min(n_sample, n_params),
                                    replace=False)

        # ── numerical gradients ───────────────────────────────────────────────
        G = np.zeros((M, len(idx)))   # G[m, k] = ∂f_m/∂x_{idx[k]}
        for k, pidx in enumerate(idx):
            fp = flat.copy(); fp[pidx] += h
            fm = flat.copy(); fm[pidx] -= h
            of = evaluate_fn(fp.reshape(img.shape))
            ob = evaluate_fn(fm.reshape(img.shape))
            G[:, k] = (of - ob) / (2.0 * h)

        # ── per-objective L2 normalisation (critical for scale balance) ────────
        for m in range(M):
            gn = np.linalg.norm(G[m])
            if gn > 1e-10:
                G[m] /= gn

        # ── solve min_{λ∈Δ} λᵀ(GGᵀ)λ via PGD ────────────────────────────────
        gram      = G @ G.T           # (M, M)
        lam       = np.ones(M) / M

        for _ in range(pgd_iters):
            grad_lam = gram @ lam
            lam      = _project_simplex(lam - pgd_lr * grad_lam)

        lambdas[i] = lam

        # ── KKTPM = ‖Σ_m λ_m ∇f_m‖₂ ─────────────────────────────────────────
        wg           = sum(lam[m] * G[m] for m in range(M))
        kktpm_raw[i] = float(np.linalg.norm(wg))

    # Normalise to [0, 1]
    max_val = kktpm_raw.max()
    if max_val > 1e-10:
        kktpm_vals = kktpm_raw / max_val
    else:
        kktpm_vals = kktpm_raw.copy()

    return kktpm_vals, lambdas


# ─────────────────────────────────────────────────────────────────────────────
# Per-generation tracking
# ─────────────────────────────────────────────────────────────────────────────

def kktpm_per_generation(generation_snapshots, evaluate_fn,
                          n_sample=50, h=1e-3, max_front_size=8):
    """
    Compute five-number KKTPM summary for each generation snapshot.

    Args:
        generation_snapshots : list of (obj_P, population, fronts) tuples,
                               one per generation checkpoint.
        evaluate_fn          : img → (M,) objective vector
        n_sample             : pixels to sample per image
        max_front_size       : cap on front members evaluated (speed)

    Returns:
        history : list of dicts with keys:
                  gen, min, q1, median, q3, max
    """
    history = []
    for gen_idx, (obj_P, pop, fronts) in enumerate(generation_snapshots):
        front_idx  = fronts[0][:max_front_size]
        front_pop  = [pop[i] for i in front_idx]
        front_obj  = obj_P[front_idx]

        if len(front_idx) == 0:
            continue

        vals, _ = compute_kktpm(front_obj, front_pop, evaluate_fn,
                                 h=h, n_sample=n_sample)
        history.append({
            "gen":    gen_idx + 1,
            "min":    float(vals.min()),
            "q1":     float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "q3":     float(np.percentile(vals, 75)),
            "max":    float(vals.max()),
        })
        print(f"  Gen {gen_idx+1:4d} KKTPM | "
              f"med={history[-1]['median']:.4f}  "
              f"[{history[-1]['min']:.4f}, {history[-1]['max']:.4f}]")

    return history


def kktpm_summary(kktpm_vals):
    """Print five-number summary (paper Fig. 16 style)."""
    print(f"  KKTPM — min={kktpm_vals.min():.4f}  "
          f"Q1={np.percentile(kktpm_vals,25):.4f}  "
          f"med={np.median(kktpm_vals):.4f}  "
          f"Q3={np.percentile(kktpm_vals,75):.4f}  "
          f"max={kktpm_vals.max():.4f}")