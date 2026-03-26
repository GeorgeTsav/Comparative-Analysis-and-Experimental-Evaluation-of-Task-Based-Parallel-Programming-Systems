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
    exts = {".ppm", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".JPEG"}
    all_files = []
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            ext = os.path.splitext(name)[1]
            if ext in exts:
                all_files.append(os.path.join(dirpath, name))
    return sorted(all_files)


def gabor_kernel(lmbda, theta, psi, sigma, gamma):
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


def process_image(payload):
    file_path, cfg = payload

    with Image.open(file_path) as im:
        if im.mode != "L":
            im = im.convert("L")
        im = im.resize((cfg["resize_n"], cfg["resize_n"]))
        arr = np.asarray(im, dtype=np.float64) / 255.0

    e = gabor_energy(arr, cfg["n_scales"], cfg["n_orients"], cfg["reps"])
    h = energy_hist(e, cfg["n_bins"])

    return float(np.sum(h * np.arange(1, cfg["n_bins"] + 1, dtype=np.float64)))


def main():
    parser = argparse.ArgumentParser(description="Heavy image pipeline torcpy")
    parser.add_argument("--images", default=os.path.join(os.path.dirname(__file__), "..", "images"))
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--scales", type=int, default=5)
    parser.add_argument("--orients", type=int, default=8)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--bins", type=int, default=64)
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

    print(f"Found {len(files)} images")
    print(
        f"cfg: resize={cfg['resize_n']}, bank={cfg['n_scales']}x{cfg['n_orients']}, "
        f"reps={cfg['reps']}, bins={cfg['n_bins']}"
    )

    t0 = time.time()
    payloads = [(f, cfg) for f in files]
    out = torc.map(process_image, payloads)
    dt = time.time() - t0

    out = np.asarray(out, dtype=np.float64)
    sum_out = float(np.sum(out))

    print("=" * 60)
    print("Framework: torcpy (app01_improc_parfor)")
    print(f"Elapsed time: {dt:.6f} s")
    print(f"Reduction: sum(out)={sum_out:.6e}  mean(out)={sum_out / max(1, len(out)):.6e}")
    print("First 10 outputs:")
    m = min(10, len(out))
    for i in range(m):
        print(f"  idx={i + 1:3d}, task_out={out[i]:.6e}")
    print("=" * 60)


if __name__ == "__main__":
    torc.start(main)
