
# reconstruction/baselines.py
import numpy as np
from scipy.ndimage import gaussian_filter
# from reconstruction.pet_forward import poisson_ll

def em_reconstruction(sinogram, A, n_iter=200, scatter=None):
    """
    EM/MLEM for Poisson likelihood (standard reference baseline).
    Returns the image after n_iter iterations.
    """
    img_size = int(np.sqrt(A.shape[1]))
    x = np.ones(A.shape[1])   # uniform initialisation
    y = sinogram.flatten()
    s = scatter if scatter is not None else np.zeros_like(y)
    At1 = A.T @ np.ones(A.shape[0])   # sensitivity image

    for _ in range(n_iter):
        y_bar = A @ x + s
        y_bar = np.clip(y_bar, 1e-10, None)
        ratio = y / y_bar
        x = x * (A.T @ ratio) / (At1 + 1e-10)
        x = np.clip(x, 0, None)

    return x.reshape(img_size, img_size)


def em_with_smoothing(sinogram, A, n_iter=200, fwhm_range=(2, 8, 0.5)):
    """Sweep post-smoothing FWHM; return list of (fwhm, image) pairs."""
    base = em_reconstruction(sinogram, A, n_iter)
    results = []
    fwhm_vals = np.arange(*fwhm_range)
    for fwhm in fwhm_vals:
        sigma = fwhm / 2.355
        smoothed = gaussian_filter(base, sigma=sigma)
        results.append((fwhm, smoothed))
    return results


def map_reconstruction(sinogram, A, beta_vals=(0.1, 0.5, 1, 5, 10, 50),
                       tol=1e-3, max_iter=500):
    """
    MAP with quadratic penalty (Section III-F).
    Returns list of (beta, image) pairs.
    """
    img_size = int(np.sqrt(A.shape[1]))
    results = []
    for beta in beta_vals:
        x = np.ones(A.shape[1])
        y = sinogram.flatten()
        At1 = A.T @ np.ones(A.shape[0])

        for _ in range(max_iter):
            y_bar = np.clip(A @ x, 1e-10, None)
            grad_ll = A.T @ (y / y_bar - 1)
            # Quadratic penalty gradient: beta * 2x (simplified)
            grad_pen = beta * 2 * x
            # Gradient ascent step (maximise LL - penalty)
            step = 1.0 / (At1 + 2*beta + 1e-10)
            x_new = np.clip(x + step * (grad_ll - grad_pen), 0, None)
            if np.mean(np.abs(x_new - x) / (np.abs(x) + 1e-10)) < tol:
                break
            x = x_new

        results.append((beta, x.reshape(img_size, img_size)))
    return results