"""
Heavy Monte Carlo basket option with barrier (torcpy version)

Requires:
- numpy
- torcpy
"""

import time
import os
import argparse

# Restrict native math libraries to one thread per process/task.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torcpy as torc


def make_spd_corr(n, alpha, rng):
    """Creates random symmetric positive definite correlation matrix.
    Blends identity (uncorrelated) with random correlation."""
    a = rng.standard_normal((n, n))
    g = a @ a.T
    d = np.sqrt(np.diag(g))
    c = g / np.outer(d, d)
    c = (1.0 - alpha) * np.eye(n) + alpha * c
    c = 0.5 * (c + c.T) + 1e-8 * np.eye(n)
    return c


def mc_batch(payload):
    """Task wrapper for torcpy.map(): unpacks and runs Monte Carlo batch simulation.
    Returns: (sum of payoffs, sum of squared payoffs, path count) for stats aggregation."""
    n_paths, seed, cfg, lmat = payload

    rng = np.random.default_rng(seed)

    n_assets = cfg["n_assets"]
    n_steps = cfg["n_steps"]
    dt = cfg["dt"]
    r = cfg["r"]
    sigma = cfg["sigma"]
    barrier = cfg["barrier"]

    s = np.ones((n_assets, n_paths), dtype=np.float64)
    alive = np.ones((n_paths,), dtype=bool)

    mu = (r - 0.5 * sigma * sigma) * dt
    sd = sigma * np.sqrt(dt)

    for _ in range(n_steps):
        z = rng.standard_normal((n_assets, n_paths))
        dw = lmat @ z

        s *= np.exp(mu + sd * dw)
        s *= (1.0 + 0.01 * np.sin(s) + 0.01 * np.log1p(np.abs(s)))

        alive &= np.all(s > barrier, axis=0)

    basket = np.mean(s, axis=0)
    k = 1.0
    pay = np.maximum(basket - k, 0.0)
    pay[~alive] = 0.0

    pay *= np.exp(-r * n_steps * dt)

    out_sum = float(np.sum(pay))
    out_sumsq = float(np.sum(pay * pay))
    out_n = int(pay.size)

    return out_sum, out_sumsq, out_n


def main():
    """Main driver: distributes Monte Carlo batches via torcpy.map() for parallel pricing."""
    parser = argparse.ArgumentParser(description="Heavy Monte Carlo torcpy")
    parser.add_argument("--paths", type=int, default=3200000)
    parser.add_argument("--batch", type=int, default=50000)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--assets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = {
        "n_steps": args.steps,
        "n_assets": args.assets,
        "dt": 1.0 / 252.0,
        "r": 0.02,
        "sigma": 0.30,
        "barrier": 0.60,
    }

    rng = np.random.default_rng(args.seed)
    c = make_spd_corr(cfg["n_assets"], 0.15, rng)
    lmat = np.linalg.cholesky(c)

    n_tasks = (args.paths + args.batch - 1) // args.batch

    print(
        f"MC torcpy: paths={args.paths} batch={args.batch} tasks={n_tasks} "
        f"steps={cfg['n_steps']} assets={cfg['n_assets']}"
    )

    t0 = time.time()

    payloads = []
    for k in range(n_tasks):
        n_this = min(args.batch, args.paths - k * args.batch)
        seed = args.seed + k + 1
        payloads.append((n_this, seed, cfg, lmat))

    out = torc.map(mc_batch, payloads)

    s1 = 0.0
    s2 = 0.0
    n = 0
    for out_sum, out_sumsq, out_n in out:
        s1 += out_sum
        s2 += out_sumsq
        n += out_n

    mean_payoff = s1 / n
    var_payoff = max(0.0, (s2 / n) - mean_payoff * mean_payoff)
    stderr = np.sqrt(var_payoff / n)

    elapsed = time.time() - t0

    print("=" * 60)
    print("Framework: torcpy (app02_mc)")
    print(
        f"Payoff mean={mean_payoff:.6g}  stdErr={stderr:.3g}  "
        f"Elapsed time={elapsed:.3f}s  throughput={n / elapsed:.3g} paths/s"
    )
    print("=" * 60)


if __name__ == "__main__":
    torc.start(main)
