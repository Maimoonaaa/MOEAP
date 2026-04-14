# optimizer/moeap.py
import numpy as np
from copy import deepcopy

# ── Nondominated sorting (NSGA-II fast sort) ──────────────────────────────────

def nondominated_sort(obj_values):
    """
    obj_values: (N, M) array — we MAXIMISE all objectives.
    Returns list of fronts, each front is a list of indices.
    """
    N = len(obj_values)
    dominated_by = [[] for _ in range(N)]
    domination_count = np.zeros(N, dtype=int)
    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # i dominates j if better or equal in all, strictly better in one
            if np.all(obj_values[i] >= obj_values[j]) and \
               np.any(obj_values[i] >  obj_values[j]):
                dominated_by[i].append(j)
            elif np.all(obj_values[j] >= obj_values[i]) and \
                 np.any(obj_values[j] >  obj_values[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(obj_values, front):
    """Crowding distance for individuals in a front."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    distances = np.zeros(n)
    for m in range(obj_values.shape[1]):
        vals = obj_values[front, m]
        order = np.argsort(vals)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        rng_m = vals[order[-1]] - vals[order[0]] + 1e-12
        for k in range(1, n-1):
            distances[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / rng_m
    return distances


# ── Genetic operators ─────────────────────────────────────────────────────────

def simulated_binary_crossover(parent1, parent2, eta_c=20, p_cross=0.95):
    """SBX on flattened image arrays (paper uses eta_c=20, Pc=0.95)."""
    if np.random.rand() > p_cross:
        return parent1.copy(), parent2.copy()
    u = np.random.rand(*parent1.shape)
    beta = np.where(u <= 0.5,
                    (2*u)**(1/(eta_c+1)),
                    (1/(2*(1-u)))**(1/(eta_c+1)))
    c1 = 0.5 * ((1+beta)*parent1 + (1-beta)*parent2)
    c2 = 0.5 * ((1-beta)*parent1 + (1+beta)*parent2)
    return c1, c2


def directed_mutation(image, sinogram, system_matrix, strength=0.01):
    """
    Directed mutation: gradient step toward higher Poisson LL,
    then add small Gaussian noise. Mimics paper's directed mutation.
    """
    grad = poisson_ll_gradient(image, sinogram, system_matrix)
    mutated = image + strength * grad
    mutated += np.random.randn(*image.shape) * strength * 0.1
    mutated = np.clip(mutated, 0, None)  # non-negative emission
    return mutated


# ── Poisson log-likelihood and gradient ───────────────────────────────────────

def poisson_ll(image, sinogram, system_matrix, scatter_randoms=None):
    """
    Eq. 3 from paper: sum_i( -y_bar_i + y_i * log(y_bar_i) )
    system_matrix: (M, N) sparse forward projector (simplified 2D here)
    """
    img_flat = image.flatten()
    y_bar = system_matrix @ img_flat
    if scatter_randoms is not None:
        y_bar = y_bar + scatter_randoms
    y_bar = np.clip(y_bar, 1e-10, None)
    y = sinogram.flatten()
    return np.sum(-y_bar + y * np.log(y_bar))


def poisson_ll_gradient(image, sinogram, system_matrix, scatter_randoms=None):
    img_flat = image.flatten()
    y_bar = system_matrix @ img_flat
    if scatter_randoms is not None:
        y_bar = y_bar + scatter_randoms
    y_bar = np.clip(y_bar, 1e-10, None)
    y = sinogram.flatten()
    grad = system_matrix.T @ (y / y_bar - 1)
    return grad.reshape(image.shape)


# ── CNN objective evaluation (wraps PyTorch models) ──────────────────────────

def evaluate_cnn_objectives(image, cnn_models, device, img_size=64):
    """
    Evaluate all CNN models on a single image.
    Returns dict of objective values (to MAXIMISE).
    """
    import torch
    img_norm = image / (image.max() + 1e-8)
    tensor = torch.tensor(
        img_norm[None, None].astype(np.float32)).to(device)
    results = {}
    with torch.no_grad():
        for name, model in cnn_models.items():
            results[name] = model(tensor).item()
    return results


# ── Main MOEAP loop (Algorithm 1 from paper) ──────────────────────────────────

class MOEAP:
    def __init__(self, sinogram, system_matrix,
                 cnn_models, device,
                 objectives=("poisson_ll", "nsnr", "inv_rmse"),
                 pop_size=50,
                 max_gen=100,
                 img_size=64,
                 eta_c=20, p_cross=0.95):

        self.sinogram = sinogram
        self.A = system_matrix
        self.cnn_models = cnn_models
        self.device = device
        self.objectives = objectives
        self.N = pop_size
        self.max_gen = max_gen
        self.H = self.W = img_size
        self.eta_c = eta_c
        self.p_cross = p_cross

        # Initialise population around FBP (uniform noise centred at FBP)
        fbp = self._fbp_init()
        self.population = [
            np.clip(fbp + np.random.uniform(-0.1, 0.1, fbp.shape), 0, None)
            for _ in range(self.N)
        ]

        self.obj_history = []   # track Pareto front per generation

    def _fbp_init(self):
        """Simple backprojection as initialisation."""
        y = self.sinogram.flatten()
        bp = self.A.T @ y
        return (bp / (bp.max() + 1e-8)).reshape(self.H, self.W)

    def _evaluate(self, img):
        vals = []
        for obj in self.objectives:
            if obj == "poisson_ll":
                vals.append(poisson_ll(img, self.sinogram, self.A))
            else:
                cnn_vals = evaluate_cnn_objectives(
                    img, self.cnn_models, self.device)
                vals.append(cnn_vals.get(obj, 0.0))
        return np.array(vals)

    def _evaluate_population(self, pop):
        return np.array([self._evaluate(img) for img in pop])

    def _select_parents(self, obj_values, fronts):
        """Binary tournament selection by rank, then crowding distance."""
        N = self.N
        rank = np.zeros(len(obj_values), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        selected = []
        for _ in range(N):
            a, b = np.random.choice(len(obj_values), 2, replace=False)
            if rank[a] < rank[b]:
                selected.append(a)
            elif rank[b] < rank[a]:
                selected.append(b)
            else:
                selected.append(a if np.random.rand() < 0.5 else b)
        return selected

    def run(self, verbose=True):
        # Evaluate initial population
        P = self.population
        obj_P = self._evaluate_population(P)
        fronts = nondominated_sort(obj_P)

        for gen in range(1, self.max_gen + 1):
            # Select parents, create offspring via SBX + directed mutation
            parent_idx = self._select_parents(obj_P, fronts)
            Q = []
            for i in range(0, self.N, 2):
                p1 = P[parent_idx[i]].flatten()
                p2 = P[parent_idx[min(i+1, self.N-1)]].flatten()
                c1, c2 = simulated_binary_crossover(p1, p2, self.eta_c, self.p_cross)
                c1 = directed_mutation(c1.reshape(self.H, self.W),
                                       self.sinogram, self.A).flatten()
                c2 = directed_mutation(c2.reshape(self.H, self.W),
                                       self.sinogram, self.A).flatten()
                Q.append(c1.reshape(self.H, self.W))
                Q.append(c2.reshape(self.H, self.W))
            Q = Q[:self.N]

            # Evaluate offspring
            obj_Q = self._evaluate_population(Q)

            # Combine parent + offspring (size 2N), nondominated sort
            R = P + Q
            obj_R = np.vstack([obj_P, obj_Q])
            fronts_R = nondominated_sort(obj_R)

            # Select next generation (Algorithm 1, lines 9-21)
            new_P, new_obj = [], []
            for front in fronts_R:
                if len(new_P) + len(front) <= self.N:
                    for idx in front:
                        new_P.append(R[idx])
                        new_obj.append(obj_R[idx])
                else:
                    # Fill remaining slots using crowding distance
                    needed = self.N - len(new_P)
                    cd = crowding_distance(obj_R, front)
                    sorted_front = sorted(zip(cd, front),
                                          reverse=True)[:needed]
                    for _, idx in sorted_front:
                        new_P.append(R[idx])
                        new_obj.append(obj_R[idx])
                    break

            P = new_P
            obj_P = np.array(new_obj)
            fronts = nondominated_sort(obj_P)
            self.obj_history.append(obj_P[fronts[0]])

            if verbose and gen % 10 == 0:
                front0 = obj_P[fronts[0]]
                print(f"  Gen {gen:4d} | front size={len(fronts[0])} "
                      f"| obj means={front0.mean(axis=0).round(3)}")

        self.population = P
        self.obj_values = obj_P
        self.fronts = fronts
        return P, obj_P, fronts