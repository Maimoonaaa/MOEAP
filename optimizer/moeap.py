# optimizer/moeap.py
"""
MOEAP with per-generation KKTPM snapshot storage.

Key changes over baseline:
  • self.generation_snapshots stores (obj_P, pop_copy, fronts) every
    `kktpm_every` generations — used by kktpm_per_generation().
  • Gradient normalisation in directed_mutation prevents step explosion.
  • Hypervolume indicator computed at each generation for an additional
    convergence metric.
  • obj_history stores full population objectives (not just front) so
    the visualiser can plot the complete cloud.
"""

import numpy as np
from copy import deepcopy
from optimizer.kktpm import compute_kktpm

# ─────────────────────────────────────────────────────────────────────────────
# Non-dominated sort  (NSGA-II fast sort)
# ─────────────────────────────────────────────────────────────────────────────

def nondominated_sort(obj_values):
    """
    Maximise all objectives.
    Returns list of fronts, each a list of population indices.
    """
    N = len(obj_values)
    dominated_by    = [[] for _ in range(N)]
    domination_count = np.zeros(N, dtype=int)
    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (np.all(obj_values[i] >= obj_values[j]) and
                    np.any(obj_values[i] >  obj_values[j])):
                dominated_by[i].append(j)
            elif (np.all(obj_values[j] >= obj_values[i]) and
                      np.any(obj_values[j] >  obj_values[i])):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


# ─────────────────────────────────────────────────────────────────────────────
# Crowding distance
# ─────────────────────────────────────────────────────────────────────────────

def crowding_distance(obj_values, front):
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    distances = np.zeros(n)
    for m in range(obj_values.shape[1]):
        vals  = obj_values[front, m]
        order = np.argsort(vals)
        distances[order[0]]  = np.inf
        distances[order[-1]] = np.inf
        rng_m = vals[order[-1]] - vals[order[0]] + 1e-12
        for k in range(1, n - 1):
            distances[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / rng_m
    return distances


# ─────────────────────────────────────────────────────────────────────────────
# Hypervolume (2-D only, exact)
# ─────────────────────────────────────────────────────────────────────────────

def hypervolume_2d(obj_values, front, ref=None):
    """
    Exact 2-D hypervolume for front (maximisation).
    ref: reference point (default: per-dim minimum of all solutions).
    """
    pts = obj_values[front]
    if pts.shape[1] != 2:
        return np.nan
    if ref is None:
        ref = obj_values.min(axis=0) - 1e-6
    # Sort by first objective descending
    order = np.argsort(pts[:, 0])[::-1]
    pts   = pts[order]
    hv    = 0.0
    prev_y = ref[1]
    for p in pts:
        if p[0] > ref[0] and p[1] > prev_y:
            hv     += (p[0] - ref[0]) * (p[1] - prev_y)
            prev_y  = p[1]
    return hv


# ─────────────────────────────────────────────────────────────────────────────
# Genetic operators
# ─────────────────────────────────────────────────────────────────────────────

def simulated_binary_crossover(parent1, parent2, eta_c=20, p_cross=0.95):
    if np.random.rand() > p_cross:
        return parent1.copy(), parent2.copy()
        
    # 50% chance for Spatial Patch Crossover (anatomy-preserving)
    if np.random.rand() < 0.5 and len(parent1.shape) == 2:
        H, W = parent1.shape
        c1, c2 = parent1.copy(), parent2.copy()
        x1 = np.random.randint(0, W - W//4)
        y1 = np.random.randint(0, H - H//4)
        pw = np.random.randint(W//8, W//2)
        ph = np.random.randint(H//8, H//2)
        
        patch_p1 = parent1[y1:y1+ph, x1:x1+pw].copy()
        patch_p2 = parent2[y1:y1+ph, x1:x1+pw].copy()
        c1[y1:y1+ph, x1:x1+pw] = patch_p2
        c2[y1:y1+ph, x1:x1+pw] = patch_p1
        return c1, c2

    u    = np.random.rand(*parent1.shape)
    beta = np.where(u <= 0.5,
                    (2*u)**(1/(eta_c+1)),
                    (1/(2*(1-u)))**(1/(eta_c+1)))
    c1 = 0.5 * ((1+beta)*parent1 + (1-beta)*parent2)
    c2 = 0.5 * ((1-beta)*parent1 + (1+beta)*parent2)
    return c1, c2


def poisson_ll(image, sinogram, system_matrix, scatter_randoms=None):
    img_flat = image.flatten()
    y_bar    = system_matrix @ img_flat
    if scatter_randoms is not None:
        y_bar = y_bar + scatter_randoms
    y_bar = np.clip(y_bar, 1e-10, None)
    y     = sinogram.flatten()
    return float(np.sum(-y_bar + y * np.log(y_bar)))


def poisson_ll_gradient(image, sinogram, system_matrix, scatter_randoms=None):
    img_flat = image.flatten()
    y_bar    = system_matrix @ img_flat
    if scatter_randoms is not None:
        y_bar = y_bar + scatter_randoms
    y_bar = np.clip(y_bar, 1e-10, None)
    y     = sinogram.flatten()
    grad  = system_matrix.T @ (y / y_bar - 1)
    return grad.reshape(image.shape)


def directed_mutation(image, sinogram, system_matrix, strength=0.01):
    """
    Gradient step in the Poisson-LL direction + TV smoothing + Multiplicative noise.
    """
    from scipy.ndimage import gaussian_filter
    
    grad      = poisson_ll_gradient(image, sinogram, system_matrix)
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-8:
        grad = grad / grad_norm
        
    # Edge-preserving TV-like proxy: diffuse local noise
    diffused  = gaussian_filter(image, sigma=1.0)
    tv_pull   = diffused - image
    tv_norm   = np.linalg.norm(tv_pull)
    if tv_norm > 1e-8:
        tv_pull = tv_pull / tv_norm
        
    # 80% Poisson LL ascent, 20% Structural TV diffusion
    mutated  = image + strength * (0.8 * grad + 0.2 * tv_pull)
    
    # Introduce small multiplicative noise so dark areas stay clean
    noise = np.random.uniform(0.95, 1.05, image.shape)
    mutated = mutated * noise
    
    upper = max(5.0, image.mean() * 10)
    return np.clip(mutated, 0.0, upper)


def evaluate_cnn_objectives(image, cnn_models, device, img_size=64):
    import torch
    img_norm = image / (image.max() + 1e-8)
    tensor   = torch.tensor(
        img_norm[None, None].astype(np.float32)).to(device)
    results  = {}
    with torch.no_grad():
        for name, model in cnn_models.items():
            results[name] = model(tensor).item()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MOEAP
# ─────────────────────────────────────────────────────────────────────────────

class MOEAP:
    def __init__(self, sinogram, system_matrix,
                 cnn_models, device,
                 objectives=("poisson_ll", "nsnr", "inv_rmse"),
                 pop_size=50, max_gen=100, img_size=64,
                 eta_c=20, p_cross=0.95,
                 kktpm_every=10, kktpm_front_size=5):
        """
        kktpm_every       : compute KKTPM every N generations
        kktpm_front_size  : number of front-0 members used for KKTPM
        """
        self.sinogram      = sinogram
        self.A             = system_matrix
        self.cnn_models    = cnn_models
        self.device        = device
        self.objectives    = list(objectives)
        self.N             = pop_size
        self.max_gen       = max_gen
        self.H = self.W    = img_size
        self.eta_c         = eta_c
        self.p_cross       = p_cross
        self.kktpm_every   = kktpm_every
        self.kktpm_front_sz = kktpm_front_size

        # History containers
        self.obj_history          = []   # list of (N_p, M) arrays per gen
        self.kktpm_history        = []   # scalar per checkpoint gen
        self.kktpm_full_history   = []   # five-number per checkpoint gen
        self.hv_history           = []   # hypervolume per gen (2-obj only)
        self.generation_snapshots = []   # (obj_P, pop, fronts) for external use

        seeds = self._fbp_init()
        self.population = []
        for i in range(self.N):
            base_seed = seeds[i % len(seeds)]
            # Multiplicative noise to preserve zero boundaries smoothly
            noise = np.random.uniform(0.9, 1.1, base_seed.shape)
            member = np.clip(base_seed * noise, 0.0, None)
            self.population.append(member)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fbp_init(self):
        from reconstruction.baselines import em_reconstruction
        y = self.sinogram.flatten()
        
        bp = self.A.T @ y
        fbp = (bp / (bp.max() + 1e-8)).reshape(self.H, self.W)
        
        em_2 = em_reconstruction(y, self.A, n_iter=2)
        em_5 = em_reconstruction(y, self.A, n_iter=5)
        em_10 = em_reconstruction(y, self.A, n_iter=10)
        
        return [fbp, em_2, em_5, em_10]

    def _evaluate(self, img):
        vals = []
        for obj in self.objectives:
            if obj == "poisson_ll":
                vals.append(poisson_ll(img, self.sinogram, self.A))
            else:
                cnn_v = evaluate_cnn_objectives(img, self.cnn_models, self.device)
                vals.append(cnn_v.get(obj, 0.0))
        return np.array(vals, dtype=float)

    def _evaluate_population(self, pop):
        return np.array([self._evaluate(img) for img in pop])

    def _select_parents(self, obj_values, fronts):
        rank = np.zeros(len(obj_values), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        selected = []
        for _ in range(self.N):
            a, b = np.random.choice(len(obj_values), 2, replace=False)
            if rank[a] < rank[b]:
                selected.append(a)
            elif rank[b] < rank[a]:
                selected.append(b)
            else:
                selected.append(a if np.random.rand() < 0.5 else b)
        return selected

    def _compute_kktpm_checkpoint(self, obj_P, P, fronts):
        fi  = fronts[0][:self.kktpm_front_sz]
        fp  = [P[i] for i in fi]
        fo  = obj_P[fi]
        k, _ = compute_kktpm(fo, fp, self._evaluate,
                              h=1e-3, n_sample=50)
        self.kktpm_history.append(float(np.median(k)))
        self.kktpm_full_history.append({
            "min":    float(k.min()),
            "q1":     float(np.percentile(k, 25)),
            "median": float(np.median(k)),
            "q3":     float(np.percentile(k, 75)),
            "max":    float(k.max()),
        })

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, verbose=True):
        P      = self.population
        obj_P  = self._evaluate_population(P)
        fronts = nondominated_sort(obj_P)

        for gen in range(1, self.max_gen + 1):

            # ── create offspring ───────────────────────────────────────────────
            parent_idx = self._select_parents(obj_P, fronts)
            Q = []
            for i in range(0, self.N, 2):
                p1 = P[parent_idx[i]].reshape(self.H, self.W)
                p2 = P[parent_idx[min(i+1, self.N-1)]].reshape(self.H, self.W)
                c1, c2 = simulated_binary_crossover(p1, p2, self.eta_c, self.p_cross)
                c1 = directed_mutation(c1, self.sinogram, self.A)
                c2 = directed_mutation(c2, self.sinogram, self.A)
                Q.append(c1); Q.append(c2)
            Q = Q[:self.N]

            # ── evaluate and combine ───────────────────────────────────────────
            obj_Q  = self._evaluate_population(Q)
            R      = P + Q
            obj_R  = np.vstack([obj_P, obj_Q])
            front_R = nondominated_sort(obj_R)

            # ── environmental selection ────────────────────────────────────────
            new_P, new_obj = [], []
            for front in front_R:
                if len(new_P) + len(front) <= self.N:
                    for idx in front:
                        new_P.append(R[idx])
                        new_obj.append(obj_R[idx])
                else:
                    needed = self.N - len(new_P)
                    cd     = crowding_distance(obj_R, front)
                    top    = sorted(zip(cd, front), reverse=True)[:needed]
                    for _, idx in top:
                        new_P.append(R[idx])
                        new_obj.append(obj_R[idx])
                    break

            P      = new_P
            obj_P  = np.array(new_obj)
            fronts = nondominated_sort(obj_P)

            # ── track history ──────────────────────────────────────────────────
            self.obj_history.append(obj_P.copy())

            # Hypervolume (2-obj only)
            if obj_P.shape[1] == 2:
                hv = hypervolume_2d(obj_P, fronts[0])
                self.hv_history.append(hv)

            # KKTPM checkpoint
            if gen % self.kktpm_every == 0:
                self._compute_kktpm_checkpoint(obj_P, P, fronts)
                self.generation_snapshots.append(
                    (obj_P.copy(), list(P), [list(f) for f in fronts])
                )

            if verbose and gen % 10 == 0:
                f0 = obj_P[fronts[0]]
                print(f"  Gen {gen:4d} | front={len(fronts[0])} "
                      f"| obj_mean={f0.mean(axis=0).round(3)}"
                      + (f" | HV={self.hv_history[-1]:.2f}"
                         if self.hv_history else "")
                      + (f" | KKTPM_med={self.kktpm_history[-1]:.4f}"
                         if self.kktpm_history else ""))

        self.population = P
        self.obj_values = obj_P
        self.fronts     = fronts
        return P, obj_P, fronts