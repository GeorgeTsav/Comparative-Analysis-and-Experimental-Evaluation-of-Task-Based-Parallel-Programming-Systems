"""
=============================================================
generate_per_app_figures.py
=============================================================
Παράγει τις ίδιες γραφικές παραστάσεις με το generate_all_figures.py
αλλά ΧΩΡΙΣΤΑ για κάθε εφαρμογή — ένα αρχείο ανά app.

Αρχεία εξόδου (στον φάκελο plots_per_app/):

  fig1_app00_framework_comparison.png/pdf   ← CMA-ES
  fig1_app01_framework_comparison.png/pdf   ← Gabor
  fig1_app02_framework_comparison.png/pdf   ← Monte Carlo

  fig2_app00_numa_effect.png/pdf
  fig2_app01_numa_effect.png/pdf
  fig2_app02_numa_effect.png/pdf

  fig3_app00_torcpy_heatmap_standard.png/pdf
  fig3_app01_torcpy_heatmap_standard.png/pdf
  fig3_app02_torcpy_heatmap_standard.png/pdf

  fig3b_app00_torcpy_heatmap_numa.png/pdf
  fig3b_app01_torcpy_heatmap_numa.png/pdf
  fig3b_app02_torcpy_heatmap_numa.png/pdf

  fig4_app00_efficiency.png/pdf
  fig4_app01_efficiency.png/pdf
  fig4_app02_efficiency.png/pdf

Δομή φακέλων που απαιτείται:
  data/
      benchmark_starpupy_20260617_222820.csv
      benchmark_starpupy_20260622_184951.csv
      benchmark_torcpy_20260617_232152.csv
      benchmark_torcpy_20260622_194311.csv
  plots_per_app/   (δημιουργείται αυτόματα)

Εξαρτήσεις:
  pip install matplotlib pandas numpy

Χρήση:
  python generate_per_app_figures.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================
# 0. ΦΑΚΕΛΟΣ ΕΞΟΔΟΥ
# =============================================================
os.makedirs("plots_per_app", exist_ok=True)


# =============================================================
# 1. ΣΤΑΘΕΡΕΣ & ΣΤΥΛ
# =============================================================

COLOR_TORCPY   = "#1b6ca8"
COLOR_STARPUPY = "#d4651c"
COLOR_STANDARD = "#5b8c5a"
COLOR_NUMA     = "#9b3d3d"

APP_ORDER = ["app00", "app01", "app02"]

APP_TITLES = {
    "app00": "App00: CMA-ES\n(Poisson PDE, sync-bound)",
    "app01": "App01: Gabor Filter Bank\n(image texture, memory-bound)",
    "app02": "App02: Monte Carlo\n(basket option, compute-bound)",
}

APP_LABELS = {
    "app00": "App00 - CMA-ES (Poisson PDE)",
    "app01": "App01 - Gabor Filter Bank",
    "app02": "App02 - Monte Carlo (Basket Option)",
}

APP_SHORT = {
    "app00": "app00",
    "app01": "app01",
    "app02": "app02",
}

CORE_TICKS = [1, 2, 4, 8, 16, 32, 64]

FILES = {
    ("starpupy", "standard"):   "data/benchmark_starpupy_20260617_222820.csv",
    ("starpupy", "numa_aware"): "data/benchmark_starpupy_20260622_184951.csv",
    ("torcpy",   "standard"):   "data/benchmark_torcpy_20260617_232152.csv",
    ("torcpy",   "numa_aware"): "data/benchmark_torcpy_20260622_194311.csv",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "lines.linewidth":   2.0,
        "lines.markersize":  6,
    })


# =============================================================
# 2. ΦΟΡΤΩΣΗ & ΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# =============================================================

def _app_code(app_name):
    return app_name.split("_")[0]


def load_raw():
    frames = []
    for (framework, numa_mode), path in FILES.items():
        df = pd.read_csv(path)
        df["framework"] = framework
        if "processes" not in df.columns:
            df["processes"] = 1
        df["app"]         = df["app_name"].apply(_app_code)
        df["app_label"]   = df["app"].map(APP_LABELS)
        df["total_cores"] = df["processes"] * df["workers"]
        df["time_s"]      = df["min_time_seconds"]
        frames.append(df[["framework", "numa_mode", "app", "app_label",
                           "processes", "workers", "total_cores", "time_s", "status"]])
    return pd.concat(frames, ignore_index=True)


def add_speedup_efficiency(df, group_cols, time_col="time_s"):
    df = df.copy()
    base = (df[df["total_cores"] == 1]
            .set_index(group_cols)[time_col]
            .rename("t_base"))
    df = df.join(base, on=group_cols)
    df["speedup"]    = df["t_base"] / df[time_col]
    df["efficiency"] = df["speedup"] / df["total_cores"]
    return df


def best_per_core(df):
    idx = df.groupby(
        ["framework", "numa_mode", "app", "app_label", "total_cores"]
    )["time_s"].idxmin()
    best = df.loc[idx].reset_index(drop=True)
    best = add_speedup_efficiency(best, group_cols=["framework", "numa_mode", "app"])
    return best.sort_values(["framework", "numa_mode", "app", "total_cores"])


def _nice_time_ticks(ymax):
    if ymax <= 20:
        step = 5
    elif ymax <= 60:
        step = 10
    elif ymax <= 120:
        step = 20
    else:
        step = 50
    return list(range(0, int(ymax) + step, step))


def _save(fig, base_name):
    """Αποθηκεύει PNG και PDF στον φάκελο plots_per_app/."""
    fig.savefig(f"plots_per_app/{base_name}.png", bbox_inches="tight")
    fig.savefig(f"plots_per_app/{base_name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================
# 3. FIG 1 ανά app — χρόνος εκτέλεσης & speedup
# =============================================================

def plot_fig1_per_app(best):
    """
    Για κάθε app: ένα figure με 2 panels (πάνω=χρόνος, κάτω=speedup).
    Ίδιο layout με το αρχικό fig1 αλλά μόνο για μία εφαρμογή.
    """
    std = best[best["numa_mode"] == "standard"]

    for app in APP_ORDER:
        fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        for fw, color, marker in [
            ("torcpy",   COLOR_TORCPY,   "o"),
            ("starpupy", COLOR_STARPUPY, "s"),
        ]:
            sub = (std[(std["app"] == app) & (std["framework"] == fw)]
                   .sort_values("total_cores"))
            if sub.empty:
                continue
            label = "torcpy" if fw == "torcpy" else "StarPUPy"
            ax_t.plot(sub["total_cores"], sub["time_s"],
                      marker=marker, color=color, label=label)
            ax_s.plot(sub["total_cores"], sub["speedup"],
                      marker=marker, color=color, label=label)

        # Ιδανική speedup
        ideal_x = np.array(CORE_TICKS)
        ax_s.plot(ideal_x, ideal_x,
                  linestyle=":", color="gray", linewidth=1.4, label="Ideal")

        # Άξονας χρόνου — LINEAR
        ymax   = std[std["app"] == app]["time_s"].max()
        yticks = _nice_time_ticks(ymax)
        ax_t.set_yticks(yticks)
        ax_t.set_yticklabels([str(t) for t in yticks])
        ax_t.set_ylim(0, yticks[-1] * 1.05)
        ax_t.set_ylabel("Execution time (s)")
        ax_t.legend(loc="upper right", framealpha=0.9)

        # Άξονας speedup — log2
        ax_s.set_xscale("log", base=2)
        ax_s.set_yscale("log", base=2)
        ax_s.set_xticks(CORE_TICKS)
        ax_s.set_xticklabels(CORE_TICKS)
        ax_s.set_yticks(CORE_TICKS)
        ax_s.set_yticklabels(CORE_TICKS)
        ax_s.set_xlabel("Total cores (best config)")
        ax_s.set_ylabel("Speedup vs. 1 core")
        ax_s.legend(loc="upper left", framealpha=0.9)

        fig.suptitle(
            f"torcpy vs StarPUPy: Execution Time & Speedup\n"
            f"{APP_TITLES[app].replace(chr(10), ' — ')}\n"
            f"(Standard Mode, athena3)",
            fontsize=12,
        )
        fig.tight_layout()
        _save(fig, f"fig1_{app}_framework_comparison")
        print(f"  ✓  fig1_{app}")


# =============================================================
# 4. FIG 2 ανά app — NUMA effect
# =============================================================

def plot_fig2_per_app(best):
    """
    Για κάθε app: ένα figure με 2 panels (πάνω=torcpy, κάτω=StarPUPy).
    Κάθε panel: standard vs numa_aware.
    """
    for app in APP_ORDER:
        fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

        for ax, fw_key, fw_label in [
            (ax_t, "torcpy",   "torcpy"),
            (ax_s, "starpupy", "StarPUPy"),
        ]:
            for numa_mode, color, marker, label in [
                ("standard",   COLOR_STANDARD, "o", "Standard"),
                ("numa_aware", COLOR_NUMA,     "^", "NUMA-aware"),
            ]:
                sub = (best[
                    (best["framework"] == fw_key) &
                    (best["numa_mode"] == numa_mode) &
                    (best["app"] == app)
                ].sort_values("total_cores"))
                if sub.empty:
                    continue
                ax.plot(sub["total_cores"], sub["time_s"],
                        marker=marker, color=color, label=label)

            # LINEAR y-axis
            ymax   = best[(best["framework"] == fw_key) & (best["app"] == app)]["time_s"].max()
            yticks = _nice_time_ticks(ymax)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks])
            ax.set_ylim(0, yticks[-1] * 1.05)
            ax.set_xscale("log", base=2)
            ax.set_xticks(CORE_TICKS)
            ax.set_xticklabels(CORE_TICKS)
            ax.set_ylabel(f"{fw_label}\nExecution time (s)")
            ax.legend(loc="upper right", framealpha=0.9)

        ax_s.set_xlabel("Total cores (best config)")

        fig.suptitle(
            f"NUMA-Aware vs Standard Execution\n"
            f"{APP_TITLES[app].replace(chr(10), ' — ')}\n"
            f"(athena3)",
            fontsize=12,
        )
        fig.tight_layout()
        _save(fig, f"fig2_{app}_numa_effect")
        print(f"  ✓  fig2_{app}")


# =============================================================
# 5. FIG 3 & 3b ανά app — torcpy heatmap
# =============================================================

def _draw_heatmap_single(raw, app, numa_mode, out_base, title):
    """
    Heatmap για μία μόνο εφαρμογή — ένα panel, μεγαλύτερο και πιο ευανάγνωστο.
    """
    subset    = raw[(raw["framework"] == "torcpy") & (raw["numa_mode"] == numa_mode)]
    best_all  = best_per_core(raw)
    torc_best = best_all[(best_all["framework"] == "torcpy") &
                         (best_all["numa_mode"] == numa_mode)]

    proc_vals   = sorted(subset["processes"].unique())
    worker_vals = sorted(subset["workers"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sub  = subset[subset["app"] == app]
    grid = sub.pivot(index="processes", columns="workers", values="time_s")
    grid = grid.reindex(index=proc_vals, columns=worker_vals)

    im = ax.imshow(np.log10(grid.values), cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(len(worker_vals)))
    ax.set_xticklabels(worker_vals)
    ax.set_yticks(range(len(proc_vals)))
    ax.set_yticklabels(proc_vals)
    ax.set_xlabel("Workers (threads) per process", fontsize=12)
    ax.set_ylabel("MPI processes", fontsize=12)
    ax.grid(False)

    # Λεξικό: (processes, workers) → total_cores για τα best-config κελιά
    best_app  = torc_best[torc_best["app"] == app]
    star_cells = {
        (int(row.processes), int(row.workers)): int(row.total_cores)
        for _, row in best_app.iterrows()
    }

    flat_min = np.nanmin(grid.values)

    for i, p in enumerate(proc_vals):
        for j, w in enumerate(worker_vals):
            val = grid.values[i, j]
            if np.isnan(val):
                continue

            is_min  = np.isclose(val, flat_min)
            is_star = (int(p), int(w)) in star_cells

            # Χρόνος — πάνω μέρος κελιού
            ax.text(
                j, i - 0.18, f"{val:.1f}",
                ha="center", va="center",
                fontsize=8,
                color="white" if is_min else "black",
                fontweight="bold" if is_min else "normal",
            )

            # ★ Nc — κάτω μέρος κελιού
            if is_star:
                tc = star_cells[(int(p), int(w))]
                ax.text(
                    j, i + 0.25, f"★ {tc}c",
                    ha="center", va="center",
                    fontsize=7.5,
                    color="white" if is_min else "#c0392b",
                    fontweight="bold",
                )

            # Κόκκινο πλαίσιο για global minimum
            if is_min:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.5,
                ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(time, s)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="$★$", color="w", markerfacecolor="#c0392b",
               markersize=11, linestyle="None",
               label="★ Nc  =  best config for N total cores (x-axis of Figs 1, 2 & 4)"),
        Patch(facecolor="none", edgecolor="red", linewidth=2,
              label="Red box  =  global best config for this application"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=1,
              fontsize=9, framealpha=0.95)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out_base)


def plot_fig3_per_app(raw):
    for app in APP_ORDER:
        _draw_heatmap_single(
            raw, app,
            numa_mode="standard",
            out_base=f"fig3_{app}_torcpy_heatmap_standard",
            title=(
                f"torcpy: MPI-Process × Thread Execution Time (Standard Mode)\n"
                f"{APP_TITLES[app].replace(chr(10), ' — ')}"
            ),
        )
        print(f"  ✓  fig3_{app}")


def plot_fig3b_per_app(raw):
    for app in APP_ORDER:
        _draw_heatmap_single(
            raw, app,
            numa_mode="numa_aware",
            out_base=f"fig3b_{app}_torcpy_heatmap_numa",
            title=(
                f"torcpy: MPI-Process × Thread Execution Time (NUMA-Aware Mode)\n"
                f"{APP_TITLES[app].replace(chr(10), ' — ')}"
            ),
        )
        print(f"  ✓  fig3b_{app}")


# =============================================================
# 6. FIG 4 ανά app — Parallel Efficiency
# =============================================================

def plot_fig4_per_app(best):
    """
    Για κάθε app: ένα figure με έναν άξονα efficiency (%).
    """
    std = best[best["numa_mode"] == "standard"]

    for app in APP_ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        for fw, color, marker, label in [
            ("torcpy",   COLOR_TORCPY,   "o", "torcpy"),
            ("starpupy", COLOR_STARPUPY, "s", "StarPUPy"),
        ]:
            sub = (std[(std["app"] == app) & (std["framework"] == fw)]
                   .sort_values("total_cores"))
            if sub.empty:
                continue
            ax.plot(sub["total_cores"], sub["efficiency"] * 100,
                    marker=marker, color=color, label=label)

        ax.axhline(100, linestyle=":", color="gray", linewidth=1.4, label="Ideal (100%)")

        ax.set_xscale("log", base=2)
        ax.set_xticks(CORE_TICKS)
        ax.set_xticklabels(CORE_TICKS)
        ax.set_ylim(0, 115)
        ax.set_xlabel("Total cores (best config)")
        ax.set_ylabel("Parallel efficiency (%)")
        ax.legend(loc="upper right", framealpha=0.9)

        fig.suptitle(
            f"Parallel Efficiency: torcpy vs StarPUPy\n"
            f"{APP_TITLES[app].replace(chr(10), ' — ')}\n"
            f"(Standard Mode, athena3)",
            fontsize=12,
        )
        fig.tight_layout()
        _save(fig, f"fig4_{app}_efficiency")
        print(f"  ✓  fig4_{app}")


# =============================================================
# 7. MAIN
# =============================================================

if __name__ == "__main__":
    set_style()

    print("Loading data...")
    raw  = load_raw()
    best = best_per_core(raw)
    print(f"  Raw rows : {len(raw)}")
    print(f"  Best rows: {len(best)}")
    print()

    print("Fig 1 — Execution time & Speedup per app:")
    plot_fig1_per_app(best)

    print("\nFig 2 — NUMA effect per app:")
    plot_fig2_per_app(best)

    print("\nFig 3 — torcpy heatmap (standard) per app:")
    plot_fig3_per_app(raw)

    print("\nFig 3b — torcpy heatmap (NUMA-aware) per app:")
    plot_fig3b_per_app(raw)

    print("\nFig 4 — Parallel efficiency per app:")
    plot_fig4_per_app(best)

    print()
    print("All figures saved in plots_per_app/")
    print("  PNG  →  plots_per_app/fig*_app0*.png")
    print("  PDF  →  plots_per_app/fig*_app0*.pdf")
