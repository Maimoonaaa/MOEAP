# optimizer/kktpm.py
import numpy as np

def compute_kktpm(obj_values, population, evaluate_fn, h=1e-3):
    """
    KKTPM: measures closeness to Pareto-optimal front.
    values should decrease toward 0 as solutions converge.
    """
    N, M = obj_values.shape
    kktpm_vals = np.zeros(N)

    for i, img in enumerate(population):
        flat = img.flatten()
        n_params = len(flat)

        # Sample a small number of parameters for speed
        sample_idx = np.random.choice(n_params,
                                      size=min(50, n_params),
                                      replace=False)
        grads = np.zeros((M, len(sample_idx)))

        for k, pidx in enumerate(sample_idx):
            flat_plus  = flat.copy(); flat_plus[pidx]  += h
            flat_minus = flat.copy(); flat_minus[pidx] -= h
            obj_plus  = evaluate_fn(flat_plus.reshape(img.shape))
            obj_minus = evaluate_fn(flat_minus.reshape(img.shape))
            grads[:, k] = (obj_plus - obj_minus) / (2 * h)
        for m in range(M):
            gn = np.linalg.norm(grads[m])
            if gn > 1e-10:
                grads[m] /= gn

        G = grads @ grads.T
        lambda_vec = np.ones(M) / M
        lr = 0.05
        for _ in range(300):
            grad_lambda = G @ lambda_vec
            lambda_vec = lambda_vec - lr * grad_lambda
            lambda_vec = _project_simplex(lambda_vec)

        # KKTPM=norm of weighted gradient sum (should → 0 at optimum)
        weighted_grad = np.zeros(len(sample_idx))
        for m in range(M):
            weighted_grad += lambda_vec[m] * grads[m]
        kktpm_vals[i] = float(np.linalg.norm(weighted_grad))

    #Normalise to [0, 1]
    max_val = kktpm_vals.max()
    if max_val > 1e-10:
        kktpm_vals /= max_val
    return kktpm_vals


def _project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def kktpm_summary(kktpm_vals):
    print(f"  KKTPM — min={kktpm_vals.min():.4f}  "
          f"Q1={np.percentile(kktpm_vals,25):.4f}  "
          f"med={np.median(kktpm_vals):.4f}  "
          f"Q3={np.percentile(kktpm_vals,75):.4f}  "
          f"max={kktpm_vals.max():.4f}")