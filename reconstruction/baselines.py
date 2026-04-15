import numpy as np
from scipy.ndimage import gaussian_filter

def em_reconstruction(sinogram, A, n_iter=200, scatter=None):
    
    img_size = int(np.sqrt(A.shape[1]))
    x = np.ones(A.shape[1])
    y = sinogram.flatten()
    s = scatter if scatter is not None else np.zeros_like(y)
    At1 = A.T @ np.ones(A.shape[0])

    for _ in range(n_iter):
        y_bar = A @ x + s
        y_bar = np.clip(y_bar, 1e-10, None)
        ratio = y / y_bar
        x = x * (A.T @ ratio) / (At1 + 1e-10)
        x = np.clip(x, 0, None)

    return x.reshape(img_size, img_size)


def em_with_smoothing(sinogram, A, n_iter=200, fwhm_range=(2, 8, 0.5)):
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
    img_size = int(np.sqrt(A.shape[1]))
    results = []
    for beta in beta_vals:
        x = np.ones(A.shape[1])
        y = sinogram.flatten()
        At1 = A.T @ np.ones(A.shape[0])

        for _ in range(max_iter):
            y_bar = np.clip(A @ x, 1e-10, None)
            grad_ll = A.T @ (y / y_bar - 1)
            grad_pen = beta * 2 * x
            step = 1.0 / (At1 + 2*beta + 1e-10)
            x_new = np.clip(x + step * (grad_ll - grad_pen), 0, None)
            if np.mean(np.abs(x_new - x) / (np.abs(x) + 1e-10)) < tol:
                break
            x = x_new

        results.append((beta, x.reshape(img_size, img_size)))
    return results