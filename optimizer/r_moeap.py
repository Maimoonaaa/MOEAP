# optimizer/r_moeap.py
import numpy as np
from optimizer.moeap import MOEAP, nondominated_sort, simulated_binary_crossover, directed_mutation

class RMOEAP(MOEAP):

    def __init__(self, *args, reference_points=None, epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_points = reference_points or []
        self.epsilon = epsilon

    def _proximity_to_refs(self, obj_vec):
        """Minimum Euclidean distance from obj_vec to any reference point."""
        if not self.ref_points:
            return 0.0
        #Normalise each objective to [0,1] range
        dists = [np.linalg.norm(obj_vec - rp) for rp in self.ref_points]
        return min(dists)

    def _select_from_front(self, obj_R, front, needed):
        
        proxs = [(self._proximity_to_refs(obj_R[i]), i) for i in front]
        proxs.sort(key=lambda x: x[0])

        selected = []
        accepted_vecs = []
        for prox, idx in proxs:
            if len(selected) >= needed:
                break
            too_close = any(
                np.linalg.norm(obj_R[idx] - av) < self.epsilon
                for av in accepted_vecs
            )
            if not too_close:
                selected.append(idx)
                accepted_vecs.append(obj_R[idx])
        if len(selected) < needed:
            for prox, idx in proxs:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= needed:
                    break

        return selected[:needed]

    def run(self, verbose=True):
        P = self.population
        obj_P = self._evaluate_population(P)
        fronts = nondominated_sort(obj_P)

        for gen in range(1, self.max_gen + 1):
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
            obj_Q = self._evaluate_population(Q)

            R = P + Q
            obj_R = np.vstack([obj_P, obj_Q])
            fronts_R = nondominated_sort(obj_R)

            #Fill next generation—use ref-point proximity in last front
            new_P, new_obj = [], []
            for front in fronts_R:
                if len(new_P) + len(front) <= self.N:
                    for idx in front:
                        new_P.append(R[idx])
                        new_obj.append(obj_R[idx])
                else:
                    needed = self.N - len(new_P)
                    chosen = self._select_from_front(obj_R, front, needed)
                    for idx in chosen:
                        new_P.append(R[idx])
                        new_obj.append(obj_R[idx])
                    break

            P = new_P
            obj_P = np.array(new_obj)
            fronts = nondominated_sort(obj_P)

            if verbose and gen % 10 == 0:
                front0 = obj_P[fronts[0]]
                print(f"  Gen {gen:4d} | front={len(fronts[0])} "
                      f"| obj means={front0.mean(axis=0).round(3)}")

        self.population = P
        self.obj_values = obj_P
        self.fronts = fronts
        return P, obj_P, fronts