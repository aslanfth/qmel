# Equitable Online Meta-Learning for AI-RAN Services (QMeL)

Code for the QMeL experiments: two-user regression, K-user regression
scalability, multi-user Rainbow MNIST, and shared-model CelebA. The
K-user scalability implementation in this repository uses the
bounded-perturbation experiment and closed-form FTL weights.

## Installation

Use Python 3.10 or newer. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install PyTorch and torchvision for your hardware before running the neural-network
experiments. The original experiments were developed on an Apple M1 Pro system
with 16 GB RAM; runtime measurements will vary across systems.

## Experiments

### K-user regression scalability

`linear-regression-K-user.py` is the **current** K-user scalability experiment.
It uses bounded perturbations decaying as `1/t`, an exact scalar meta-gradient,
closed-form follow-the-leader (FTL) user weights, and a realized-sequence
hindsight comparator optimized for each time prefix.
`common.py` provides its shared numerical and aggregation helpers.

Quick verification (small horizon and configuration grid):

```bash
python linear-regression-K-user.py --verify
```

Run the manuscript experiment configuration (50,000 rounds, 10 random seeds,
`K = 4, 16, 64, 256`, and `alpha = 1, 2`):

```bash
python linear-regression-K-user.py
```

Results are saved to `results/k_scalability/full/`, including:

- `trajectories/`: one CSV per random seed and configuration;
- `aggregated/k{K}_alpha{alpha}.csv`: trajectories averaged over seeds;
- `summary_per_seed.csv` and `summary_aggregated.csv`: end-of-run statistics;
- `config.json`: experiment settings and software information.

Plotting CSVs contain `round`, `regret_mean`, and `regret_std`, among other fields.
Use `--save-user-histories` to additionally save per-user trajectories as `.npz`
files (these may be large). Use `--help` to see all options; `--overwrite`
replaces only directories created by the experiment's own output marker.

### Two-user regression

`linear-regression-2-user.py` implements the separate two-user regression
comparison, including fixed-weight baselines. Run it from the repository root:

```bash
python linear-regression-2-user.py
```

It writes a sampled comparison CSV to `paper_results/`.

### Rainbow MNIST

`multiuser-mnist.py` contains the multi-user visual-inference experiment.
It requires the MNIST dataset, PyTorch, torchvision, and `higher`.
Review the data paths and training configuration in the script before running.

### CelebA

`experiments/celeba/` contains the shared-model fair multi-task learning
comparison, with training code and method implementations under `methods/`.
See `experiments/celeba/trainer.py`, `trainer_equity.py`, and `run.sh` for the
configuration and dataset setup. The CelebA comparison disables inner
adaptation (`eta=0`) to match the shared-model baselines.

## Reproducibility and generated data

Generated CSVs, downloaded datasets, model checkpoints, and training logs are
not included in this source repository. `results/`, `paper_results/`, and
experiment checkpoint/log directories are excluded by `.gitignore`.
