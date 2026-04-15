import numpy as np
from scipy.sparse import csr_matrix

def make_system_matrix(img_size=64, n_angles=60, n_detectors=None):

    if n_detectors is None:
        n_detectors = img_size
    N = img_size * img_size
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    rows, cols, data = [], [], []
    M_total = 0
    cx = cy = img_size / 2.0

    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for d in range(n_detectors):
            offset = d - n_detectors/2 + 0.5
            t_vals = np.linspace(-img_size/2, img_size/2, img_size*2)
            xs = cx + t_vals*cos_a - offset*sin_a
            ys = cy + t_vals*sin_a + offset*cos_a
            xi = np.floor(xs).astype(int)
            yi = np.floor(ys).astype(int)
            valid = (xi >= 0) & (xi < img_size) & \
                    (yi >= 0) & (yi < img_size)
            pixel_ids = xi[valid] + yi[valid]*img_size
            for pid in np.unique(pixel_ids):
                rows.append(M_total)
                cols.append(pid)
                data.append(1.0)
            M_total += 1

    A = csr_matrix((data, (rows, cols)), shape=(M_total, N))

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    A = A.multiply(1.0 / row_sums[:, None])
    return A


def simulate_sinogram(true_image, A, n_events=8e4, scatter_frac=0.1):
    y_mean = A @ true_image.flatten()
    y_mean = y_mean / (y_mean.sum() + 1e-10) * n_events
    scatter = scatter_frac * y_mean.mean() * np.ones_like(y_mean)
    y_mean_total = y_mean + scatter
    y_noisy = np.random.poisson(y_mean_total).astype(float)
    return y_noisy.reshape(A.shape[0]), scatter

