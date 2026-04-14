# optimizer/kktpm.py
import numpy as np

def compute_kktpm(obj_values, population, evaluate_fn, h=1e-4):
    """
    Karush-Kuhn-Tucker Proximity Measure (Section III-C).
    Measures closeness of solutions to the Pareto-optimal front.

    For each solution x in a nondominated front:
      KKTPM(x) = min over lambda>=0, ||lambda||=1 of ||grad_L||
    where L = -sum_m lambda_m * f_m(x) is the Lagrangian.

    We approximate gradients numerically with finite differences
    on the flattened image vector.

    Args:
        obj_values : (N, M) array of objective values
        population : list of N images (H, W arrays)
        evaluate_fn: callable, takes image → (M,) objective array
        h          : finite difference step

    Returns:
        kktpm_vals: (N,) array of KKTPM values, range [0, 1]
    """
    N, M = obj_values.shape
    kktpm_vals = np.zeros(N)

    for i, img in enumerate(population):
        flat = img.flatten()
        n_params = len(flat)

        # Numerical gradient for each objective — sub-sample for speed
        # Sample at most 500 random parameters per image
        sample_idx = np.random.choice(n_params,
                                      size=min(500, n_params),
                                      replace=False)
        grads = np.zeros((M, len(sample_idx)))

        for k, pidx in enumerate(sample_idx):
            flat_plus = flat.copy(); flat_plus[pidx] += h
            flat_minus = flat.copy(); flat_minus[pidx] -= h
            obj_plus  = evaluate_fn(flat_plus.reshape(img.shape))
            obj_minus = evaluate_fn(flat_minus.reshape(img.shape))
            grads[:, k] = (obj_plus - obj_minus) / (2*h)

        # Find optimal lambda (convex hull projection)
        # Min ||sum_m lambda_m * grad_m||^2, lambda>=0, sum=1
        # Solve via projected gradient descent on lambda simplex
        G = grads @ grads.T    # (M, M)
        lambda_vec = np.ones(M) / M

        for _ in range(200):
            grad_lambda = G @ lambda_vec
            lambda_vec = lambda_vec - 0.01 * grad_lambda
            # Project onto probability simplex
            lambda_vec = _project_simplex(lambda_vec)

        # KKTPM = ||grad L|| normalised
        weighted_grad = sum(lambda_vec[m] * grads[m] for m in range(M))
        kktpm_vals[i] = np.linalg.norm(weighted_grad) / (np.linalg.norm(grads) + 1e-8)

    return kktpm_vals


def _project_simplex(v):
    """Project vector v onto the probability simplex sum=1, v>=0."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def kktpm_summary(kktpm_vals):
    """Print the 5-number summary used in Fig. 16 of the paper."""
    print(f"  KKTPM — min={kktpm_vals.min():.4f}  "
          f"Q1={np.percentile(kktpm_vals,25):.4f}  "
          f"med={np.median(kktpm_vals):.4f}  "
          f"Q3={np.percentile(kktpm_vals,75):.4f}  "
          f"max={kktpm_vals.max():.4f}")