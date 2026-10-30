#!/usr/bin/env python3
"""Reproducible K-user QMeL regression scalability experiment.

Uses bounded O(1/t) perturbations, the exact scalar meta-gradient,
closed-form FTL dual weights, and a numerical hindsight comparator
for each horizon prefix. Comparator evaluation is excluded from the
recorded learning-update runtime. See README.md for usage and outputs.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import (
    aggregate_trajectories, alpha_fairness, assert_finite,
    first_crossing, parse_csv_list, prepare_output,
    sustained_crossing, write_json,
)


def generate_data(T: int, K: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate bounded uniform perturbations decaying as 1/t."""    
    rng = np.random.default_rng(seed)
    radius = 0.1 * np.sqrt(3.0)
    time = np.arange(1, T + 1, dtype=np.float64)
    a = np.clip(1.0 + rng.uniform(-radius, radius, T) / time, 0.8, 1.0)
    base = np.linspace(0.0, 1.0, K, dtype=np.float64)
    y = np.clip(base + rng.uniform(-radius, radius, (T, K)) / time[:, None], 0.0, 1.0)
    return a, y


def exact_prefix_comparator(a: np.ndarray, y: np.ndarray, alpha: float,
                            eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve 1D hindsight optimization for every prefix of the *realized* tasks."""
    T, K = y.shape
    ybar = np.mean(y, axis=1)
    A = a * (1.0 - eta * a*a)
    B = eta * a[:, None]**2 * ybar[:, None] - y
    s2 = np.cumsum(0.5 * A*A)
    s1 = np.cumsum(A[:, None] * B, axis=0)
    s0 = np.cumsum(0.5 * B*B, axis=0)
    xstar = np.empty(T)
    FStar = np.empty(T)

    for t in range(T):
        n = t + 1

        def negative_equity(x: float) -> float:
            util = 1.0 - (s2[t]*x*x + s1[t]*x + s0[t]) / n
            return -float(np.sum(alpha_fairness(util, alpha)))

        opt = minimize_scalar(negative_equity, bounds=(0., 1.), method='bounded',
                              options={'xatol': 1e-11, 'maxiter': 500})
        if not opt.success:
            raise RuntimeError(f'Comparator failed for prefix {n}: {opt.message}')
        choices = [(0.0, negative_equity(0.0)),
                   (1.0, negative_equity(1.0)),
                   (float(opt.x), negative_equity(float(opt.x)))]
        bestx, neg_val = min(choices, key=lambda pair: pair[1])
        xstar[t] = bestx
        FStar[t] = -neg_val

    return xstar, FStar


def analytic_meta_gradient(x: float, a: float, y: np.ndarray,
                           w: np.ndarray, eta: float) -> float:
    theta = x - eta * np.mean(a * (a*x - y))
    return float((1.0 - eta*a*a) * np.dot(w, -a*(a*theta-y)))


def verify_derivative() -> float:
    x, a, eta = 0.37, 0.91, 0.5
    y = np.array([0.13, 0.42, 0.72, 0.96])
    w = np.array([1.4, 2.1, 1.2, 1.6])

    def objective(z: float) -> float:
        theta = z - eta*np.mean(a*(a*z-y))
        u = 1.0 - 0.5*(a*theta-y)**2
        return float(w @ u)

    eps = 1e-6
    finite_diff = (objective(x+eps)-objective(x-eps))/(2*eps)
    analytic = analytic_meta_gradient(x, a, y, w, eta)
    error = abs(analytic-finite_diff) / max(1., abs(finite_diff))
    if error > 1e-7:
        raise AssertionError(f'K-user finite-difference gradient error {error}')
    return error


def run_one(T: int, K: int, alpha: float, seed: int,
            output: Path, verify: bool, save_user_histories: bool) -> tuple[pd.DataFrame, dict]:
    a, y = generate_data(T, K, seed)
    if verify:
        a2, y2 = generate_data(T, K, seed)
        assert np.array_equal(a, a2) and np.array_equal(y, y2)
    assert np.all((a >= 0.8) & (a <= 1.0))
    assert np.all((y >= 0.) & (y <= 1.))
    eta, umin, umax = 0.5, 0.5, 1.0
    mu = 0.8**2
    outer_lr = 8 * (umax**alpha) / (mu*K)

    # Identical seed-based initialization across alpha configurations.
    x = float(np.random.default_rng(seed + 1_000_000).uniform(0., 1.))
    w = np.ones(K)
    cum_util = np.zeros(K)
    utilities = np.empty((T, K))
    weights = np.empty((T, K))
    initialization = np.empty(T)
    adapted = np.empty(T)
    xafter = np.empty(T)
    timing_us = np.empty(T)

    for t in range(T):
        tic = perf_counter_ns()
        at, yt = a[t], y[t]
        initialization[t] = x
        weights[t] = w
        theta = x - eta*np.mean(at*(at*x-yt))
        adapted[t] = theta
        u = 1.0 - 0.5*(at*theta-yt)**2
        if np.any(u < umin-1e-10) or np.any(u > umax+1e-10):
            raise AssertionError(f'K-user utility outside analytic range at t={t+1}')
        utilities[t] = u

        grad = (1.0-eta*at*at) * np.dot(w, -at*(at*theta-yt))
        x = float(np.clip(x + outer_lr*grad/(t+1), 0., 1.))
        xafter[t] = x
        cum_util += u
        w = np.clip(cum_util/(t+1), umin, umax)**(-alpha)
        timing_us[t] = (perf_counter_ns() - tic) / 1000.

    assert_finite('utility history', utilities)
    assert_finite('weight history', weights)
    cum_avg = np.cumsum(utilities, axis=0)/np.arange(1, T+1)[:, None]
    equity = np.sum(alpha_fairness(cum_avg, alpha), axis=1)
    xstar, equity_star = exact_prefix_comparator(a, y, alpha, eta)
    regret = equity_star - equity
    # A relative gap is undefined if the comparator's equity equals zero.
    gap = np.divide(100.*np.abs(regret), np.abs(equity_star),
                    out=np.full(T, np.nan), where=np.abs(equity_star)>1e-12)
    if not np.any(np.isfinite(gap)):
        raise AssertionError('No defined relative gaps')

    if verify:
        # A dense grid must not beat our numerical scalar optimizer.
        grid = np.linspace(0., 1., 1001)
        for t in (0, T//2, T-1):
            ybar = y[:t+1].mean(axis=1)
            theta_grid = (1-eta*a[:t+1, None]**2)*grid[None, :] + eta*a[:t+1, None]*ybar[:, None]
            residual = a[:t+1, None, None]*theta_grid[:, :, None]-y[:t+1, None, :]
            avg_u = (1-.5*residual**2).mean(axis=0)
            max_grid = np.max(np.sum(alpha_fairness(avg_u, alpha), axis=1))
            if equity_star[t] < max_grid-1e-8:
                raise AssertionError('Numerical hindsight optimum below brute-force grid')
        if not np.all((weights >= 1-1e-9) & (weights <= (2.**alpha)+1e-9)):
            raise AssertionError('K-user weights outside dual domain')

    df = pd.DataFrame({
        'round': np.arange(1, T+1), 'equity': equity,
        'equity_star': equity_star, 'regret': regret, 'gap_percent': gap,
        'x_star': xstar, 'x_pre': initialization, 'theta': adapted,
        'x_post': xafter, 'runtime_us': timing_us,
        'weight_min': weights.min(axis=1), 'weight_max': weights.max(axis=1),
        'utility_min': utilities.min(axis=1), 'utility_max': utilities.max(axis=1),
    })
    tag = f'k{K}_alpha{alpha:g}_seed{seed}'
    df.to_csv(output/'trajectories'/f'{tag}.csv', index=False)
    if save_user_histories:
        np.savez_compressed(output/'user_histories'/f'{tag}.npz',
                            utility=utilities, weight=weights, a=a, y=y,
                            initialization=initialization, adapted=adapted)

    metrics = {
        'alpha': alpha, 'K': K, 'seed': seed, 'T': T,
        'final_equity': float(equity[-1]), 'final_equity_star': float(equity_star[-1]),
        'final_regret': float(regret[-1]), 'final_gap_percent': float(gap[-1]),
        't10_first': first_crossing(gap, 10.),
        't5_first': first_crossing(gap, 5.),
        't10_sustained': sustained_crossing(gap, 10.),
        't5_sustained': sustained_crossing(gap, 5.),
        'mean_runtime_us': float(timing_us.mean()),
        'median_runtime_us': float(np.median(timing_us)),
        'min_utility': float(utilities.min()), 'max_utility': float(utilities.max()),
        'min_weight': float(weights.min()), 'max_weight': float(weights.max()),
        'final_x_star': float(xstar[-1]),
    }
    return df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verify', action='store_true', help='Run fast numerical checks and a small multi-seed grid')
    parser.add_argument('--T', type=int, default=50_000, help='Number of online rounds')
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--alphas', default='1,2')
    parser.add_argument('--users', default='4,16,64,256', help='Comma-separated K values')
    parser.add_argument('--save-user-histories', action='store_true',
                        help='Store all K per-user utilities/weights as .npz (large for T=50k)')
    parser.add_argument('--output', type=Path, default=None, help='Output directory')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output generated by this script')
    args = parser.parse_args()
    T = 24 if args.verify else args.T
    seeds = [0, 1] if args.verify else parse_csv_list(args.seeds, int)
    alphas = [1., 2.] if args.verify else parse_csv_list(args.alphas, float)
    users = [4, 16] if args.verify else parse_csv_list(args.users, int)
    if T < 2 or any(a <= 0 for a in alphas) or any(k < 2 for k in users) or any(s < 0 for s in seeds):
        parser.error('Require T>=2, alpha>0, K>=2, and nonnegative seeds')
    output = args.output or Path('results/k_scalability')/('verify' if args.verify else 'full')
    out = prepare_output(output, args.overwrite)
    gradient_error = verify_derivative()
    write_json(out/'config.json', {
        'experiment': 'k_scalability', 'verify': args.verify, 'T': T,
        'seeds': seeds, 'alphas': alphas, 'users': users,
        'eta': 0.5, 'u_min': 0.5, 'u_max': 1., 'mu_lower': 0.8**2,
        'noise': 'uniform radius sqrt(3)*0.1, clipped, 1/t',
        'dual_update': 'closed-form FTL; next-round weight',
        'hindsight': 'exact realized-sequence prefix scalar optimization',
        'save_user_histories': bool(args.save_user_histories or args.verify),
        'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
        'platform': platform.platform(), 'finite_difference_relative_error': gradient_error,
    })
    rows = []
    long_groups = []
    for alpha in alphas:
        for K in users:
            frames = []
            for seed in seeds:
                print(f'RUN K={K}, alpha={alpha:g}, seed={seed}, T={T}', flush=True)
                frame, metrics = run_one(T, K, alpha, seed, out, args.verify,
                                         args.save_user_histories or args.verify)
                frames.append(frame)
                rows.append(metrics)
                print(f'K={K:3d} alpha={alpha:g} seed={seed} '
                      f'final regret={metrics["final_regret"]:.6g} '
                      f'10%={metrics["t10_first"]} 5%={metrics["t5_first"]} '
                      f'runtime={metrics["mean_runtime_us"]:.1f} us')
            agg = aggregate_trajectories(frames)
            agg.insert(0, 'alpha', alpha)
            agg.insert(1, 'K', K)
            agg.to_csv(out/'aggregated'/f'k{K}_alpha{alpha:g}.csv', index=False)
            long_groups.append(agg)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'summary_per_seed.csv', index=False)
    grouped = summary.groupby(['alpha', 'K'], sort=True)
    agg_summary = grouped.agg(['mean', 'std'])
    agg_summary.columns = [f'{col}_{stat}' for col, stat in agg_summary.columns]
    counts = grouped.size().rename('n_seeds')
    agg_summary = agg_summary.join(counts).reset_index()
    agg_summary.to_csv(out/'summary_aggregated.csv', index=False)
    pd.concat(long_groups, ignore_index=True).to_csv(
        out/'aggregated'/'k_all_trajectories.csv', index=False)
    if args.verify:
        write_json(out/'verify_report.json', {
            'status': 'PASS', 'finite_difference_relative_error': gradient_error,
            'n_runs': len(rows), 'checked': [
                'deterministic data generation', 'scalar meta-gradient finite differences',
                'utility/weight bounds', 'prefix comparator vs dense grid',
                'saved full trajectories and aggregated CSVs'
            ],
        })
        print('VERIFY PASS:', out/'verify_report.json')
    print('Saved outputs under', out)


if __name__ == '__main__':
    main()
