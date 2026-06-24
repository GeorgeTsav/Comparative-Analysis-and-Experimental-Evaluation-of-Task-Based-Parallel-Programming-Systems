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

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


def source_locations(k):
    """Returns the first k fixed source point locations in domain [0,1]^2.
    Used as centers for Gaussian basis functions in PDE RHS."""
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
    """Generates m random sensor measurement locations uniformly in [0.05, 0.95]^2.
    These are observation points where solution values are sampled."""
    return rng.uniform(0.05, 0.95, size=(m, 2))


@njit(nogil=True)
def build_rhs_numba(theta, src_xy, src_sig, nx, ny):
    """Builds RHS of Poisson equation as sum of weighted Gaussian basis functions.
    Evaluates: f(x,y) = sum_k theta_k * exp(-(||p-src_k||^2)/(2*sig^2))"""
    f = np.zeros((ny, nx), dtype=np.float64)
    sig2 = src_sig * src_sig
    inv_x = 1.0 / (nx - 1)
    inv_y = 1.0 / (ny - 1)

    for k in range(theta.shape[0]):
        cx = src_xy[k, 0]
        cy = src_xy[k, 1]
        theta_k = theta[k]

        for j in range(ny):
            y = j * inv_y
            dy = y - cy
            for i in range(nx):
                x = i * inv_x
                dx = x - cx
                g = np.exp(-((dx * dx + dy * dy) / (2.0 * sig2)))
                f[j, i] += theta_k * g

    return f


@njit(nogil=True)
def solve_poisson_numba(theta, src_xy, src_sig, nx, ny, max_iter, tol, use_tol):
    """Solves Laplace equation -Delta(u) = f using Jacobi iteration on regular grid.
    Returns: (solution array, number of iterations performed)"""
    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    hx2 = hx * hx
    hy2 = hy * hy
    den = 2.0 * (1.0 / hx2 + 1.0 / hy2)

    f = build_rhs_numba(theta, src_xy, src_sig, nx, ny)
    u = np.zeros((ny, nx), dtype=np.float64)
    unew = np.zeros_like(u)

    for k in range(max_iter):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                unew[j, i] = (
                    (u[j, i + 1] + u[j, i - 1]) / hx2
                    + (u[j + 1, i] + u[j - 1, i]) / hy2
                    + f[j, i]
                ) / den

        if use_tol:
            diff_num = 0.0
            diff_den = 0.0
            for j in range(ny):
                for i in range(nx):
                    d = unew[j, i] - u[j, i]
                    diff_num += d * d
                    diff_den += unew[j, i] * unew[j, i]

            diff = np.sqrt(diff_num) / max(1e-12, np.sqrt(diff_den))
            u, unew = unew, u
            if diff < tol:
                return u, k + 1
        else:
            u, unew = unew, u

    return u, max_iter


@njit(nogil=True)
def sample_sensors_numba(u, sens_xy, nx, ny):
    """Samples solution values u at sensor locations via nearest-neighbor interpolation.
    Returns: 1D array of solution values at each sensor location."""
    vals = np.zeros((sens_xy.shape[0],), dtype=np.float64)
    x_scale = nx - 1
    y_scale = ny - 1

    for i in range(sens_xy.shape[0]):
        sx = sens_xy[i, 0]
        sy = sens_xy[i, 1]

        ix = int(sx * x_scale + 0.5)
        iy = int(sy * y_scale + 0.5)

        if ix < 0:
            ix = 0
        elif ix > nx - 1:
            ix = nx - 1

        if iy < 0:
            iy = 0
        elif iy > ny - 1:
            iy = ny - 1

        vals[i] = u[iy, ix]

    return vals


@njit(nogil=True)
def eval_candidate_numba(theta, src_xy, src_sig, sens_xy, nx, ny, max_iter, tol, use_tol, target_y):
    """Fitness function for CMA-ES: solves PDE, samples sensors, returns MSE + L2 regularization."""
    u, _ = solve_poisson_numba(theta, src_xy, src_sig, nx, ny, max_iter, tol, use_tol)
    y = sample_sensors_numba(u, sens_xy, nx, ny)

    mse = 0.0
    for i in range(target_y.shape[0]):
        d = y[i] - target_y[i]
        mse += d * d
    mse /= target_y.shape[0]

    reg = 0.0
    for i in range(theta.shape[0]):
        reg += theta[i] * theta[i]
    reg = 1e-4 * (reg / theta.shape[0])

    return mse + reg


def build_rhs(theta, pde):
    """Wrapper for build_rhs_numba: unpacks PDE config dict."""
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    return build_rhs_numba(theta, pde["src_xy"], pde["src_sig"], pde["nx"], pde["ny"])


def solve_poisson_from_theta(theta, pde):
    """Wrapper for solve_poisson_numba: unpacks PDE config and returns (solution, iterations)."""
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    return solve_poisson_numba(
        theta,
        pde["src_xy"],
        pde["src_sig"],
        pde["nx"],
        pde["ny"],
        pde["max_iter"],
        pde["tol"],
        pde["use_tol"],
    )


def sample_sensors(u, pde):
    """Wrapper for sample_sensors_numba: unpacks PDE config for sensor coordinates."""
    return sample_sensors_numba(u, pde["sens_xy"], pde["nx"], pde["ny"])


def eval_candidate(theta, pde, target_y):
    """Wrapper for eval_candidate_numba: called by StarPU task submission for fitness evaluation."""
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    target_y = np.ascontiguousarray(target_y, dtype=np.float64)
    return float(
        eval_candidate_numba(
            theta,
            pde["src_xy"],
            pde["src_sig"],
            pde["sens_xy"],
            pde["nx"],
            pde["ny"],
            pde["max_iter"],
            pde["tol"],
            pde["use_tol"],
            target_y,
        )
    )


def rmse(a, b):
    """Computes root mean square error between two arrays."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


async def main():
    """Main async driver for CMA-ES + PDE inverse problem using StarPU.
    Uses async task submission with starpu.task_submit() for distributed evaluation."""
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
        "src_xy": np.ascontiguousarray(source_locations(8), dtype=np.float64),
        "src_sig": 0.06,
        "m": 40,
        "sens_xy": np.ascontiguousarray(sensor_locations(40, rng), dtype=np.float64),
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
        # es.logger.add(es)

    elapsed = time.time() - t0

    best_x = np.asarray(es.result[0], dtype=np.float64)
    best_f = float(es.result[1])

    ubest, iters = solve_poisson_from_theta(best_x, pde)
    ybest = sample_sensors(ubest, pde)

    print("=" * 60)
    print("Framework: StarPU Python (app00_cmaes)")
    print(f"Best f: {best_f:.6e}")
    print(f"Best theta: {best_x}")
    print(f"PDE iters (best): {iters}")
    print(f"Sensor RMSE: {rmse(ybest, target_y):.3e}")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print("=" * 60)

    starpu.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
