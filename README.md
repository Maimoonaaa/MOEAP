# Evolutionary Optimization of Machine-Learned Objectives for PET Image Reconstruction

This project implements the method proposed in:

> Abouhawwash & Alessio, *"Evolutionary Optimization of Multiple Machine-Learned Objectives for PET Image Reconstruction"*, IEEE Transactions on Radiation and Plasma Medical Sciences, Vol. 7, No. 3, March 2023.

The core idea is to replace hand-crafted regularisation terms in PET image reconstruction with **CNN-predicted task performance metrics** (quantitation, detection, spatial resolution), and then optimise these learned objectives simultaneously using a **multiobjective evolutionary algorithm (NSGA-II)**. The result is a Pareto-optimal set of reconstructed images that trade off different clinical performance dimensions, rather than a single image produced by a fixed regularisation strength.

---

## Project Structure

```
pet_moeap/
├── data/
│   └── generate_dataset.py
├── models/
│   ├── checkpoints/               saved .pt files written here after training
│   └── cnn_objectives.py
├── reconstruction/
│   ├── pet_forward.py
│   └── baselines.py
├── optimizer/
│   ├── moeap.py
│   ├── r_moeap.py
│   └── kktpm.py
├── experiments/
│   └── run_experiment.py
├── plots/
│   └── visualize.py
├── results/                       all output figures written here
├── requirements.txt
└── README.md
```

---

## File Descriptions

### `data/generate_dataset.py`

Generates a synthetic PET image dataset that mimics the 9-step XCAT phantom procedure described in Section III-E of the paper. Each image is a 64×64 (paper: 128×128) 2D slice containing an elliptical organ with randomly inserted hot lesions of known location, size, and contrast. Noise is applied using a gamma distribution to reproduce the statistical properties of iteratively reconstructed PET images, followed by Gaussian smoothing with a randomly selected FWHM. Every image is labelled with ground-truth values of three figures of merit:

- **`inv_rmse`** — inverse root mean square percent error over lesion pixels (Eq. 4)
- **`nsnr`** — normalised signal-to-noise ratio (Eq. 5)
- **`inv_fwhm`** — inverse full-width-at-half-maximum of the applied smoothing filter (Eq. 6)

Output is stored as an HDF5 file (`data/pet_dataset.h5`) split 70/10/20 into train, validation, and test sets.

**Key function:** `generate_dataset(n=10_000, save_path="data/pet_dataset.h5")`

---

### `models/cnn_objectives.py`

Defines and trains three independent CNN regression models, one per figure of merit. Each model follows the architecture in Section III-E exactly: four sequential blocks of `[Conv2d(3×3) → AvgPool2d(2×2) → BatchNorm2d → ReLU]` with filter depths 8, 16, 32, 32, followed by a `Dropout(0.2)` layer and a single fully connected regression output. Input is a 64×64 single-channel image; output is a scalar prediction of the corresponding figure of merit.

Training uses SGD with momentum (lr=1e-3, momentum=0.9, weight decay=1e-4) with cosine annealing, and early stopping with patience=15 epochs on the validation MSE loss. Checkpoints are saved to `models/checkpoints/{name}_best.pt` at the best validation loss.

**Key functions:**
- `ObjectiveCNN` — PyTorch `nn.Module` defining the shared architecture
- `train_model(objective_idx, epochs)` — trains one model for the given objective index (0=inv_rmse, 1=nsnr, 2=inv_fwhm)
- `load_models(save_dir, device)` — loads all three checkpoints and returns a dict of eval-mode models
- `evaluate_all()` — computes Pearson correlation on the test set, replicating Fig. 3 of the paper

---

### `reconstruction/pet_forward.py`

Implements a simplified 2D parallel-beam PET forward model.

- **`make_system_matrix(img_size, n_angles, n_detectors)`** — constructs a sparse CSR system matrix `A` of shape `(M, N)` using ray-driven line integrals, where `N = img_size²` is the number of image pixels and `M = n_angles × n_detectors` is the number of sinogram bins. Each row is normalised to unit sum (sensitivity normalisation).
- **`simulate_sinogram(true_image, A, n_events, scatter_frac)`** — analytically projects the true image through `A`, scales to `n_events` counts, adds a uniform scatter-and-randoms term (default 10% of mean, matching Section III-F), and applies Poisson noise.
- **`poisson_ll(image, sinogram, A)`** — evaluates the Poisson log-likelihood objective function (Eq. 3), used directly as one of the MOEAP objectives.
- **`poisson_ll_gradient(image, sinogram, A)`** — computes the analytic gradient of the Poisson log-likelihood with respect to the image, used inside directed mutation.

---

### `reconstruction/baselines.py`

Implements the two conventional iterative reconstruction baselines used for comparison in Figs. 4, 6, 8, 10, 12, 13 of the paper.

- **`em_reconstruction(sinogram, A, n_iter)`** — standard expectation-maximisation (MLEM) algorithm that maximises the Poisson log-likelihood. Runs for a fixed number of iterations.
- **`em_with_smoothing(sinogram, A, n_iter, fwhm_range)`** — runs EM to convergence then sweeps post-reconstruction Gaussian smoothing over a range of FWHM values, producing a sequence of (fwhm, image) pairs that trace a curve in objective space.
- **`map_reconstruction(sinogram, A, beta_vals)`** — penalised likelihood (MAP) reconstruction with a quadratic penalty term, terminating when the per-pixel relative change falls below a tolerance. Sweeps over a list of penalty strengths `beta`, producing a sequence of (beta, image) pairs.

---

### `optimizer/moeap.py`

Core implementation of the Multiobjective Evolutionary Algorithm for PET (MOEAP), described in Algorithm 1 of the paper. The algorithm maintains a population of `N` candidate images and evolves them across `G` generations to identify the Pareto-optimal front in objective space.

- **`nondominated_sort(obj_values)`** — fast non-dominated sorting (NSGA-II) that partitions the population into fronts F₁, F₂, … where F₁ contains solutions that no other solution dominates. Maximisation is assumed for all objectives.
- **`crowding_distance(obj_values, front)`** — computes the crowding distance metric used to preserve diversity within the same front.
- **`simulated_binary_crossover(parent1, parent2, eta_c, p_cross)`** — SBX crossover operator on flattened image vectors, using distribution index `eta_c=20` and crossover probability `Pc=0.95` as in the paper.
- **`directed_mutation(image, sinogram, A, strength)`** — mutates a candidate image by taking a small normalised gradient step along the Poisson log-likelihood gradient and adding Gaussian noise, then clamping to a physically valid range.
- **`evaluate_cnn_objectives(image, cnn_models, device)`** — passes a single image through all three trained CNN models and returns a dict of predicted objective values.
- **`MOEAP`** — main class. `__init__` initialises the population around a filtered backprojection estimate. `run()` executes the generational loop: parent selection by rank and crowding distance, SBX crossover, directed mutation, combined population sorting, and truncation to size N using crowding distance on the last included front.

---

### `optimizer/r_moeap.py`

Extends `MOEAP` with the Reference Point-based method (R-NSGA-II, Section III-B), which focuses the evolutionary search on a user-specified region of the objective space rather than approximating the entire Pareto front. This is the first application of reference-point-based EMO to PET image reconstruction (as stated in Section V).

- **`RMOEAP`** — subclass of `MOEAP`. Accepts one or more reference points `r⁽ⁱ⁾` as coordinate vectors in objective space.
- **`_proximity_to_refs(obj_vec)`** — returns the minimum Euclidean distance from a solution's objective vector to any reference point.
- **`_select_from_front(obj_R, front, needed)`** — replaces the crowding-distance selection in the last front with reference-point proximity selection, enforcing an epsilon-clearance radius around each accepted solution to maintain diversity. Mimics the R-NSGA-II procedure illustrated in Fig. 1 of the paper.

---

### `optimizer/kktpm.py`

Implements the Karush–Kuhn–Tucker Proximity Measure (KKTPM) described in Section III-C, which quantifies how close a set of solutions is to the Pareto-optimal front without requiring any prior knowledge of the front. A value of 0 indicates a Pareto-optimal solution.

- **`compute_kktpm(obj_values, population, evaluate_fn, h)`** — for each solution, estimates numerical gradients of all objective functions via central finite differences over a random subsample of image pixels, then solves for the optimal KKT multipliers `λ` via projected gradient descent onto the probability simplex. KKTPM is the norm of the resulting weighted gradient sum, normalised to `[0, 1]`.
- **`_project_simplex(v)`** — projects an arbitrary vector onto the unit probability simplex (sum=1, all entries ≥ 0) using the O(n log n) algorithm.
- **`kktpm_summary(kktpm_vals)`** — prints the five-number summary (min, Q1, median, Q3, max) of KKTPM values across a front, matching the style of Fig. 16 in the paper.

---

### `experiments/run_experiment.py`

Top-level entry point that orchestrates the full pipeline. Must be run as `python -m experiments.run_experiment` from the project root (the `if __name__ == '__main__'` guard is required on Windows due to the `spawn` multiprocessing context).

Execution order:
1. Generate synthetic dataset (skipped if `data/pet_dataset.h5` already exists)
2. Check for pretrained CNN checkpoints in `models/checkpoints/`; load them if all three are present, otherwise train from scratch
3. Build the system matrix and simulate a noisy PET sinogram from a digital phantom
4. Run MOEAP for 100 generations with objectives `[Poisson LL, NSNR]`
5. Run R-MOEAP with a reference point set to the mean of the MOEAP Pareto front
6. Compute KKTPM on a 5-solution subset of the final Pareto front
7. Run EM+smoothing and MAP baseline sweeps
8. Call visualisation functions and save all figures to `results/`

---

### `plots/visualize.py`

Produces two multi-panel figures saved to `results/`.

**`plot_pareto_front(...)`** — six-panel figure containing:
- Pareto front scatter plot comparing MOEAP, R-MOEAP, EM+smoothing, and MAP in objective space
- KKTPM convergence curve across generations (should decrease toward 0)
- Histogram of objective value distributions in the final population
- EM objective tradeoff as a function of post-smoothing FWHM
- MAP objective tradeoff as a function of penalty strength β (log scale)
- Histogram of pairwise Euclidean distances between Pareto front solutions (front diversity)

**`plot_images(...)`** — three-row figure containing:
- Row 1: side-by-side reconstructed images (true, three MOEAP front solutions, EM, MAP) on a shared colour scale
- Row 2: normalised absolute error maps `|x̂ - x_true| / x_true` per method
- Row 3 left: horizontal intensity profile through the image centre for all methods
- Row 3 right: bar chart of normalised RMSE per method

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pet_moeap
cd pet_moeap

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
scipy>=1.10
h5py>=3.8
torch>=2.0
matplotlib>=3.7
scikit-learn>=1.3
```

---

## Running the Project

All commands must be run from the **project root directory** (`pet_moeap/`).

### Full pipeline (first run)

```bash
python -m experiments.run_experiment
```

This will sequentially: generate 10,000 synthetic images (~5 min), train three CNN models (~20 min on CPU, ~5 min on GPU), simulate PET data, run MOEAP and R-MOEAP for 100 generations each (~30–60 min on CPU), compute KKTPM, run baselines, and save all plots.

### Subsequent runs (checkpoints already saved)

On re-runs, the script automatically detects the saved `.pt` files in `models/checkpoints/` and skips training, going directly to the optimisation step:

```bash
python -m experiments.run_experiment
# Console output will show:
# === Pretrained checkpoints found — loading models ===
#   Found: models/checkpoints/inv_rmse_best.pt
#   Found: models/checkpoints/nsnr_best.pt
#   Found: models/checkpoints/inv_fwhm_best.pt
```

### Training CNN models only

```bash
python -m models.cnn_objectives
```

Trains all three models sequentially and saves checkpoints. Also prints Pearson correlation on the test set (replicates Fig. 3 of the paper).

### Generating the dataset only

```bash
python -m data.generate_dataset
```

---

## Expected Outputs

All outputs are written to `results/`:

| File | Description | Paper equivalent |
|---|---|---|
| `pareto_front.png` | Six-panel objective space analysis | Figs. 4, 8, 10, 12, 16 |
| `reconstruction_images.png` | Image comparison with error maps and profiles | Figs. 7, 11, 14 |
| `moeap_obj.npy` | Saved objective values for all final-generation MOEAP solutions | — |
| `moeap_pop.npy` | Saved image arrays for all final-generation MOEAP solutions | — |
| `rmoeap_obj.npy` | Saved objective values for R-MOEAP solutions | — |

---

## Differences from the Paper

| Aspect | Paper | This implementation |
|---|---|---|
| Image size | 128×128 | 64×64 |
| Training images | 50,000 | 10,000 |
| Population size | 100 | 50 |
| Generations | 200 | 100 |
| Forward projector | Philips Gemini TF scanner model | Simplified parallel-beam ray tracing |
| Data | XCAT phantom + real patient F18-FDG PET/CT | Synthetic elliptical phantom only |
| KKTPM tracking | Per generation, full front | Per generation, 5-solution subset for speed |

All algorithmic components — NSGA-II nondominated sorting, crowding distance, SBX crossover with `ηc=20`, `Pc=0.95`, directed mutation, CNN architecture (4-block, depths 8/16/32/32, dropout 0.2), reference point selection with epsilon clearance, and KKTPM via simplex projection — are implemented as described in the paper.

---

## Citation

```
@article{abouhawwash2023evolutionary,
  title={Evolutionary Optimization of Multiple Machine-Learned Objectives for PET Image Reconstruction},
  author={Abouhawwash, Mohamed and Alessio, Adam M.},
  journal={IEEE Transactions on Radiation and Plasma Medical Sciences},
  volume={7},
  number={3},
  pages={273--283},
  year={2023},
  publisher={IEEE}
}
```
