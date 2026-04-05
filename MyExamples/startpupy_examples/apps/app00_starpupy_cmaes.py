"""
CMA-ES + Poisson PDE objective (StarPU Python version)

Requires:
- numpy
- cma
- starpu python bindings
"""

import time
import asyncio
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
import cma
from starpu import starpu


def source_locations(k):
    pts = np.array([
        [0.20, 0.20],
        [0.20, 0.80],
        [0.80, 0.20],
        [0.80, 0.80],
        [0.35, 0.50],
        [0.65, 0.50],
        [0.50, 0.35],
        [0.50, 0.65],
    ])
    return pts[:k]


def sensor_locations(m, rng):
    return rng.uniform(0.05, 0.95, size=(m, 2))


def build_rhs(theta, pde):
    nx, ny = pde["nx"], pde["ny"]
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)

    src_xy = pde["src_xy"]
    sig2 = pde["src_sig"] ** 2

    f = np.zeros((ny, nx), dtype=np.float64)
    for k in range(pde["k"]):
        cx, cy = src_xy[k]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig2))
        f += theta[k] * g
    return f


def solve_poisson_from_theta(theta, pde):
    nx, ny = pde["nx"], pde["ny"]
    max_iter = pde["max_iter"]
    tol = pde["tol"]
    use_tol = pde["use_tol"]

    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    hx2 = hx * hx
    hy2 = hy * hy
    den = 2.0 * (1.0 / hx2 + 1.0 / hy2)

    f = build_rhs(theta, pde)
    u = np.zeros((ny, nx), dtype=np.float64)
    unew = np.zeros_like(u)

    iters = 0
    for k in range(max_iter):
        unew[1:-1, 1:-1] = (
            (u[1:-1, 2:] + u[1:-1, :-2]) / hx2
            + (u[2:, 1:-1] + u[:-2, 1:-1]) / hy2
            + f[1:-1, 1:-1]
        ) / den
        iters = k + 1

        if use_tol:
            diff = np.linalg.norm(unew - u) / max(1e-12, np.linalg.norm(unew))
            u, unew = unew, u
            if diff < tol:
                break
        else:
            u, unew = unew, u

    return u, iters


def sample_sensors(u, pde):
    nx, ny = pde["nx"], pde["ny"]
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    vals = np.zeros((pde["m"],), dtype=np.float64)
    for i, (sx, sy) in enumerate(pde["sens_xy"]):
        ix = np.argmin(np.abs(x - sx))
        iy = np.argmin(np.abs(y - sy))
        vals[i] = u[iy, ix]
    return vals


def eval_candidate(theta, pde, target_y):
    u, _ = solve_poisson_from_theta(theta, pde)
    y = sample_sensors(u, pde)

    mse = np.mean((y - target_y) ** 2)
    reg = 1e-4 * np.mean(theta ** 2)
    return float(mse + reg)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


async def main():
    parser = argparse.ArgumentParser(description="CMA-ES PDE StarPU")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--max-gen", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    try:
        starpu.init()
    except Exception as e:
        print(f"StarPU initialization error: {e}")
        return

    rng = np.random.default_rng(args.seed)

    pde = {
        "nx": args.nx,
        "ny": args.ny,
        "max_iter": args.max_iter,
        "tol": 1e-6,
        "use_tol": False,
        "k": 8,
        "src_xy": source_locations(8),
        "src_sig": 0.06,
        "m": 40,
        "sens_xy": sensor_locations(40, rng),
    }

    theta_true = np.array([1.2, -0.8, 0.5, 0.0, 1.6, -1.1, 0.7, -0.3], dtype=np.float64)
    target_u, _ = solve_poisson_from_theta(theta_true, pde)
    target_y = sample_sensors(target_u, pde)

    es = cma.CMAEvolutionStrategy(
        np.zeros(pde["k"]),
        0.5,
        {"popsize": args.popsize, "maxiter": args.max_gen, "seed": args.seed, "verb_disp": 0},
    )

    print(
        f"CMA-ES+PDE StarPU: n={pde['k']} pop={args.popsize} "
        f"grid={pde['nx']}x{pde['ny']} max_iter={pde['max_iter']}"
    )

    t0 = time.time()
    while not es.stop():
        solutions = es.ask()
        futures = [
            starpu.task_submit()(eval_candidate, np.asarray(s, dtype=np.float64), pde, target_y)
            for s in solutions
        ]
        fitness = [await fut for fut in futures]
        es.tell(solutions, fitness)
        es.logger.add(es)

    elapsed = time.time() - t0

    best_x = np.asarray(es.result[0], dtype=np.float64)
    best_f = float(es.result[1])

    ubest, iters = solve_poisson_from_theta(best_x, pde)
    ybest = sample_sensors(ubest, pde)

    print("=" * 60)
    print("Framework: StarPU Python (app00_cmaes_parfor)")
    print(f"Best f: {best_f:.6e}")
    print(f"Best theta: {best_x}")
    print(f"PDE iters (best): {iters}")
    print(f"Sensor RMSE: {rmse(ybest, target_y):.3e}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
