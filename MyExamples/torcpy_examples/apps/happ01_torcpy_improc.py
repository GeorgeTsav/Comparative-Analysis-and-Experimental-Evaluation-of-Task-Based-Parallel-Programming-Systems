"""
Expensive image processing pipeline (torcpy version)

Requires:
- numpy
- pillow
- torcpy
"""

import os
import time
import argparse

# Restrict native math libraries to one thread per process/task.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from PIL import Image
import torcpy as torc


def get_files(path):
    """Recursively finds image files with common extensions. Returns sorted list of paths."""
    exts = {".ppm", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".JPEG"}
    all_files = []
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            ext = os.path.splitext(name)[1]
            if ext in exts:
                all_files.append(os.path.join(dirpath, name))
    return sorted(all_files)


def gabor_kernel(lmbda, theta, psi, sigma, gamma):
    """Generates Gabor filter kernel with wavelength lmbda and orientation theta.
    Returns: zero-mean 2D Gabor filter kernel."""
    sz = max(9, 2 * int(np.ceil(3 * sigma)) + 1)
    half = sz // 2
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    k = np.exp(-(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2 * sigma ** 2))
    k *= np.cos(2 * np.pi * x_theta / lmbda + psi)
    k -= np.mean(k)
    return k


def fftconv2_same(img, ker):
    """Performs 2D convolution via FFT with output size same as input (efficient method)."""
    hi, wi = img.shape
    hk, wk = ker.shape

    h = hi + hk - 1
    w = wi + wk - 1

    fi = np.fft.fft2(img, s=(h, w))
    fk = np.fft.fft2(ker, s=(h, w))
    rfull = np.real(np.fft.ifft2(fi * fk))

    r0 = hk // 2
    c0 = wk // 2
    return rfull[r0:r0 + hi, c0:c0 + wi]


def gabor_energy(img, n_scales, n_orients, reps):
    """Computes multi-scale Gabor filter bank energy (expensive computation).
    Returns: texture energy magnitude at each pixel."""
    e = np.zeros_like(img, dtype=np.float64)

    for _ in range(reps):
        for s in range(1, n_scales + 1):
            lmbda = 2 ** (s + 1)
            sigma = 0.56 * lmbda
            gamma = 0.5
            psi = 0.0
            for o in range(n_orients):
                theta = o * np.pi / n_orients
                k = gabor_kernel(lmbda, theta, psi, sigma, gamma)
                r = fftconv2_same(img, k)
                e += r ** 2

    return np.sqrt(e + 1e-12)


def energy_hist(e, n_bins):
    """Computes histogram of log-transformed Gabor energy values."""
    x = np.log1p(e.ravel())
    mn = np.min(x)
    mx = np.max(x)
    if mx <= mn:
        h = np.zeros((n_bins,), dtype=np.float64)
        h[0] = x.size
        return h
    edges = np.linspace(mn, mx, n_bins + 1)
    h, _ = np.histogram(x, bins=edges)
    return h.astype(np.float64)


def process_image_task(file_path, cfg):
    """Child task: processes single image (second-level parallelism)."""
    with Image.open(file_path) as im:
        if im.mode != "L":
            im = im.convert("L")
        im = im.resize((cfg["resize_n"], cfg["resize_n"]))
        arr = np.asarray(im, dtype=np.float64) / 255.0

    e = gabor_energy(arr, cfg["n_scales"], cfg["n_orients"], cfg["reps"])
    h = energy_hist(e, cfg["n_bins"])

    return float(np.sum(h * np.arange(1, cfg["n_bins"] + 1, dtype=np.float64)))


def main_kernel_task(kernel_id, file_paths, cfg):
    """Top-level kernel: processes chunk of images by spawning child torcpy tasks.
    Returns: dict with timing and aggregated results for the chunk."""
    # Second level parallelism: each top-level kernel spawns image tasks.
    t0 = time.time()
    child_tasks = [torc.submit(process_image_task, f, cfg) for f in file_paths]
    torc.wait()
    out = [task.result() for task in child_tasks]
    dt = time.time() - t0

    out = np.asarray(out, dtype=np.float64)
    sum_out = float(np.sum(out))

    return {
        "kernel_id": kernel_id,
        "n_files": len(file_paths),
        "elapsed": dt,
        "sum_out": sum_out,
        "out": out,
    }


def main():
    """Main driver: nested-parallel image processing using torcpy with two task levels."""
    parser = argparse.ArgumentParser(description="Heavy image pipeline torcpy")
    parser.add_argument("--images", default=os.path.join(os.path.dirname(__file__), "..", "..", "images"))
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--scales", type=int, default=5)
    parser.add_argument("--orients", type=int, default=8)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--outer-kernels", type=int, default=2)
    args = parser.parse_args()

    cfg = {
        "resize_n": args.resize,
        "n_scales": args.scales,
        "n_orients": args.orients,
        "reps": args.reps,
        "n_bins": args.bins,
    }

    files = get_files(args.images)
    if not files:
        print(f"No images found in: {args.images}")
        return

    n_outer = max(1, args.outer_kernels)
    chunks = [list(c) for c in np.array_split(files, n_outer) if len(c) > 0]

    print(f"Found {len(files)} images")
    print(
        f"cfg: resize={cfg['resize_n']}, bank={cfg['n_scales']}x{cfg['n_orients']}, "
        f"reps={cfg['reps']}, bins={cfg['n_bins']}"
    )
    print(f"Two-level parallelism: {len(chunks)} top-level kernels x image-level child tasks")

    t0 = time.time()
    parent_tasks = [
        torc.submit(main_kernel_task, i + 1, chunk, cfg) for i, chunk in enumerate(chunks)
    ]
    torc.wait()
    parent_results = [task.result() for task in parent_tasks]
    dt = time.time() - t0

    for res in parent_results:
        print(
            f"Top kernel {res['kernel_id']}: files={res['n_files']}, "
            f"elapsed={res['elapsed']:.6f}s, sum(out)={res['sum_out']:.6e}"
        )

    out = np.concatenate([res["out"] for res in parent_results]) if parent_results else np.array([])
    sum_out = float(np.sum(out))

    print("=" * 60)
    print("Framework: torcpy (happ01_improc)")
    print(f"Elapsed time: {dt:.6f} s")
    print(f"Reduction: sum(out)={sum_out:.6e}  mean(out)={sum_out / max(1, len(out)):.6e}")
    print("First 10 outputs:")
    m = min(10, len(out))
    for i in range(m):
        print(f"  idx={i + 1:3d}, task_out={out[i]:.6e}")
    print("=" * 60)

if __name__ == "__main__":
    torc.start(main)
