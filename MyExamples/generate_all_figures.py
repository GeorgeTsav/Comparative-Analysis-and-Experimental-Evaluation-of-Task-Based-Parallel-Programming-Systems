"""
=============================================================
generate_all_figures.py
=============================================================
Παράγει και τα 5 γραφήματα της διπλωματικής εργασίας:

  fig1  — torcpy vs StarPUPy: χρόνος & speedup (standard mode)
  fig2  — NUMA effect: standard vs numa_aware, ανά framework
  fig3  — torcpy heatmap processes×workers (standard mode)
  fig3b — torcpy heatmap processes×workers (NUMA-aware mode)
  fig4  — Parallel efficiency (standard mode)

Δομή φακέλων που απαιτείται:
  data/
      benchmark_starpupy_20260617_222820.csv   ← StarPUPy standard
      benchmark_starpupy_20260622_184951.csv   ← StarPUPy NUMA-aware
      benchmark_torcpy_20260617_232152.csv     ← torcpy standard
      benchmark_torcpy_20260622_194311.csv     ← torcpy NUMA-aware
  plots/   (δημιουργείται αυτόματα αν δεν υπάρχει)

Εξαρτήσεις:
  pip install matplotlib pandas numpy

Χρήση:
  python generate_all_figures.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# =============================================================
# 0. ΔΗΜΙΟΥΡΓΙΑ ΦΑΚΕΛΟΥ ΕΞΟΔΟΥ
# =============================================================
os.makedirs("plots", exist_ok=True)


# =============================================================
# 1. ΣΤΑΘΕΡΕΣ & ΣΤΥΛ
# =============================================================

# Χρώματα — colorblind-friendly, print-friendly
COLOR_TORCPY   = "#1b6ca8"   # μπλε
COLOR_STARPUPY = "#d4651c"   # πορτοκαλί
COLOR_STANDARD = "#5b8c5a"   # πράσινο
COLOR_NUMA     = "#9b3d3d"   # κόκκινο

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

# Οι μετρημένες τιμές πυρήνων — δυνάμεις του 2
CORE_TICKS = [1, 2, 4, 8, 16, 32, 64]

# CSV αρχεία ανά (framework, numa_mode)
FILES = {
    ("starpupy", "standard"):   "data/benchmark_starpupy_20260617_222820.csv",
    ("starpupy", "numa_aware"): "data/benchmark_starpupy_20260622_184951.csv",
    ("torcpy",   "standard"):   "data/benchmark_torcpy_20260617_232152.csv",
    ("torcpy",   "numa_aware"): "data/benchmark_torcpy_20260622_194311.csv",
}

def set_style():
    """Κοινές ρυθμίσεις matplotlib για thesis-ready γραφήματα."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,   # υψηλή ανάλυση για LaTeX
        "savefig.bbox":      "tight",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   8.5,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
    })


# =============================================================
# 2. ΦΟΡΤΩΣΗ & ΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# =============================================================

def _app_code(app_name: str) -> str:
    """Εξάγει τον κωδικό εφαρμογής: 'app00_torcpy_cmaes' → 'app00'."""
    return app_name.split("_")[0]


def load_raw() -> pd.DataFrame:
    """
    Διαβάζει και τα 4 CSV και τα ενοποιεί σε ένα κοινό DataFrame με στήλες:
      framework, numa_mode, app, app_label,
      processes, workers, total_cores, time_s, status
    """
    frames = []
    for (framework, numa_mode), path in FILES.items():
        df = pd.read_csv(path)
        df["framework"] = framework
        # Το StarPUPy δεν έχει στήλη processes → προσθέτουμε processes=1
        if "processes" not in df.columns:
            df["processes"] = 1
        df["app"]         = df["app_name"].apply(_app_code)
        df["app_label"]   = df["app"].map(APP_LABELS)
        df["total_cores"] = df["processes"] * df["workers"]
        df["time_s"]      = df["min_time_seconds"]
        frames.append(df[["framework", "numa_mode", "app", "app_label",
                           "processes", "workers", "total_cores", "time_s", "status"]])
    return pd.concat(frames, ignore_index=True)


def add_speedup_efficiency(df: pd.DataFrame,
                           group_cols: list,
                           time_col: str = "time_s") -> pd.DataFrame:
    """
    Προσθέτει στήλες speedup και efficiency στο DataFrame.
      speedup    = T(total_cores=1) / T(N)
      efficiency = speedup / N        (ως ποσοστό όταν * 100)
    Ο χρόνος αναφοράς (1 core) εντοπίζεται per (framework, numa_mode, app).
    """
    df = df.copy()
    base = (
        df[df["total_cores"] == 1]
        .set_index(group_cols)[time_col]
        .rename("t_base")
    )
    df = df.join(base, on=group_cols)
    df["speedup"]    = df["t_base"] / df[time_col]
    df["efficiency"] = df["speedup"] / df["total_cores"]
    return df


def best_per_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Για κάθε (framework, numa_mode, app, total_cores) κρατά ΜΟΝΟ
    τη διαμόρφωση (processes, workers) με τον ελάχιστο χρόνο.

    Αυτό επιτρέπει δίκαιη σύγκριση torcpy (2D grid) vs StarPUPy
    (μόνο workers) στον κοινό άξονα total_cores.
    """
    idx = df.groupby(
        ["framework", "numa_mode", "app", "app_label", "total_cores"]
    )["time_s"].idxmin()
    best = df.loc[idx].reset_index(drop=True)
    best = add_speedup_efficiency(
        best, group_cols=["framework", "numa_mode", "app"]
    )
    return best.sort_values(["framework", "numa_mode", "app", "total_cores"])


# =============================================================
# 3. FIGURE 1 — torcpy vs StarPUPy: χρόνος & speedup
# =============================================================

def _nice_time_ticks(ymax: float):
    """
    Επιστρέφει clean tick values (σε δευτερόλεπτα) για γραμμικό y-axis χρόνου.
    Επιλέγει αυτόματα το βήμα ανάλογα με το εύρος τιμών.
    """
    if ymax <= 20:
        step = 5
    elif ymax <= 60:
        step = 10
    elif ymax <= 120:
        step = 20
    else:
        step = 50
    ticks = list(range(0, int(ymax) + step, step))
    return ticks


def plot_fig1(best: pd.DataFrame):
    """
    2 γραμμές × 3 στήλες:
      Πάνω:  χρόνος εκτέλεσης (s) — LINEAR y-axis, δείχνει τα πραγματικά δευτερόλεπτα
      Κάτω:  speedup             — log-log + ιδανική γραμμή
    Μόνο standard mode.
    """
    std = best[best["numa_mode"] == "standard"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

    for col, app in enumerate(APP_ORDER):
        ax_t = axes[0, col]   # άξονας χρόνου
        ax_s = axes[1, col]   # άξονας speedup

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

        # Ιδανική (γραμμική) speedup — speedup = N
        ideal_x = np.array(CORE_TICKS)
        ax_s.plot(ideal_x, ideal_x,
                  linestyle=":", color="gray", linewidth=1.2, label="Ideal")

        # --- Άξονες πάνω panel (χρόνος) — LINEAR ---
        ymax = std[std["app"] == app]["time_s"].max()
        yticks = _nice_time_ticks(ymax)

        ax_t.set_title(APP_TITLES[app])
        ax_t.set_xscale("log", base=2)
        ax_t.set_xticks(CORE_TICKS)
        ax_t.set_xticklabels(CORE_TICKS)
        ax_t.set_yticks(yticks)
        ax_t.set_yticklabels([f"{t}" for t in yticks])
        ax_t.set_ylim(0, yticks[-1] * 1.05)
        ax_t.set_ylabel("Execution time (s)" if col == 0 else "")

        # --- Άξονες κάτω panel (speedup) ---
        ax_s.set_xscale("log", base=2)
        ax_s.set_yscale("log", base=2)
        ax_s.set_xticks(CORE_TICKS)
        ax_s.set_xticklabels(CORE_TICKS)
        ax_s.set_yticks(CORE_TICKS)
        ax_s.set_yticklabels(CORE_TICKS)
        ax_s.set_xlabel("Total cores (best config)")
        ax_s.set_ylabel("Speedup vs. 1 core" if col == 0 else "")

    axes[0, 0].legend(loc="upper right", framealpha=0.9)
    axes[1, 0].legend(loc="upper left",  framealpha=0.9, fontsize=7.5)

    fig.suptitle(
        "torcpy vs StarPUPy: Execution Time and Speedup"
        " (Standard Mode, athena3, 64 cores / 128 threads)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("plots/fig1_framework_comparison_standard.png")
    fig.savefig("plots/fig1_framework_comparison_standard.pdf")
    plt.close(fig)
    print("✓  fig1 saved")


# =============================================================
# 4. FIGURE 2 — NUMA effect: standard vs numa_aware
# =============================================================

def plot_fig2(best: pd.DataFrame):
    """
    2 γραμμές × 3 στήλες:
      Πάνω:    torcpy
      Κάτω:    StarPUPy
    Κάθε panel: standard (πράσινο) vs numa_aware (κόκκινο).
    Y-άξονας: χρόνος εκτέλεσης (s), LINEAR scale — φαίνονται τα πραγματικά δευτερόλεπτα.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

    framework_rows = [("torcpy", "torcpy"), ("starpupy", "StarPUPy")]

    for row, (fw_key, fw_label) in enumerate(framework_rows):
        for col, app in enumerate(APP_ORDER):
            ax = axes[row, col]

            for numa_mode, color, marker, label in [
                ("standard",  COLOR_STANDARD, "o", "Standard"),
                ("numa_aware", COLOR_NUMA,    "^", "NUMA-aware"),
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

            # LINEAR y-axis με clean ticks
            ymax = best[(best["framework"] == fw_key) & (best["app"] == app)]["time_s"].max()
            yticks = _nice_time_ticks(ymax)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{t}" for t in yticks])
            ax.set_ylim(0, yticks[-1] * 1.05)

            ax.set_xscale("log", base=2)
            ax.set_xticks(CORE_TICKS)
            ax.set_xticklabels(CORE_TICKS)

            if row == 0:
                ax.set_title(APP_TITLES[app])
            if col == 0:
                ax.set_ylabel(f"{fw_label}\nExecution time (s)")
            if row == 1:
                ax.set_xlabel("Total cores (best config)")

    axes[0, 0].legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Effect of NUMA-Aware Execution on Benchmark Performance (athena3)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("plots/fig2_numa_effect.png")
    fig.savefig("plots/fig2_numa_effect.pdf")
    plt.close(fig)
    print("✓  fig2 saved")


# =============================================================
# 5. FIGURE 3 & 3b — torcpy heatmaps (standard & NUMA-aware)
# =============================================================

def _draw_heatmap(raw: pd.DataFrame, numa_mode: str,
                  out_png: str, out_pdf: str, title: str):
    """
    Κοινή λογική για fig3 (standard) και fig3b (numa_aware).

    Κάθε κελί (i,j) = χρόνος για proc_vals[i] processes × worker_vals[j] workers.
    Χρώμα = log10(time_s)  →  viridis_r  (κίτρινο=γρήγορο, μωβ=αργό).
    Κελιά εκτός ορίου (processes×workers > 64) εμφανίζονται κενά (NaN).
    Το βέλτιστο κελί (global minimum) σημειώνεται με κόκκινο πλαίσιο.
    Κάθε κελί που είναι το βέλτιστο ζευγάρι (processes×workers) για κάποιο
    total_cores του x-άξονα των Fig 1/2/4 σημειώνεται με ★ και τον αριθμό cores.
    """
    subset    = raw[(raw["framework"] == "torcpy") & (raw["numa_mode"] == numa_mode)]
    best_all  = best_per_core(raw)
    torc_best = best_all[(best_all["framework"] == "torcpy") &
                         (best_all["numa_mode"] == numa_mode)]

    proc_vals   = sorted(subset["processes"].unique())
    worker_vals = sorted(subset["workers"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))  # λίγο ψηλότερο για το legend

    for col, app in enumerate(APP_ORDER):
        ax  = axes[col]
        sub = subset[subset["app"] == app]

        # Μετατροπή long → 2D πίνακας
        grid = sub.pivot(index="processes", columns="workers", values="time_s")
        grid = grid.reindex(index=proc_vals, columns=worker_vals)

        # Heatmap με log10 scale στα χρώματα
        im = ax.imshow(np.log10(grid.values), cmap="viridis_r", aspect="auto")

        # Ετικέτες αξόνων
        ax.set_xticks(range(len(worker_vals)))
        ax.set_xticklabels(worker_vals)
        ax.set_yticks(range(len(proc_vals)))
        ax.set_yticklabels(proc_vals)
        ax.set_xlabel("Workers (threads) per process")
        ax.set_ylabel("MPI processes" if col == 0 else "")
        ax.set_title(APP_TITLES[app].replace("\n", " "), fontsize=9.5)
        ax.grid(False)

        # Χτίζουμε λεξικό: (processes, workers) → total_cores
        # για τα κελιά που χρησιμοποιούνται ως "best config" στον x-άξονα
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

                # Χρόνος — ελαφρώς πάνω από το κέντρο του κελιού
                ax.text(
                    j, i - 0.18, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=6.5,
                    color="white" if is_min else "black",
                    fontweight="bold" if is_min else "normal",
                )

                # ★ Nc — κάτω από τον χρόνο, μέσα στο ίδιο κελί
                if is_star:
                    tc = star_cells[(int(p), int(w))]
                    ax.text(
                        j, i + 0.25, f"★ {tc}c",
                        ha="center", va="center",
                        fontsize=6.2,
                        color="white" if is_min else "#c0392b",
                        fontweight="bold",
                    )

                # Κόκκινο πλαίσιο για το global minimum
                if is_min:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="red", linewidth=2,
                    ))

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log10(time, s)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Legend κάτω από τα panels
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker="$★$", color="w", markerfacecolor="#c0392b",
               markersize=10, linestyle="None",
               label="★ Nc  =  best (processes × workers) for N total cores"
                     " — this is the point shown at x=N in Figs 1, 2 & 4"),
        Patch(facecolor="none", edgecolor="red", linewidth=2,
              label="Red box  =  global best config for this application"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=1, fontsize=8.5, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.09))

    fig.suptitle(title, fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_fig3(raw: pd.DataFrame):
    _draw_heatmap(
        raw,
        numa_mode="standard",
        out_png="plots/fig3_torcpy_grid_heatmap.png",
        out_pdf="plots/fig3_torcpy_grid_heatmap.pdf",
        title=(
            "torcpy: Execution Time Across MPI-Process / Thread Configurations (Standard Mode)"
        ),
    )
    print("✓  fig3 saved")


def plot_fig3b(raw: pd.DataFrame):
    _draw_heatmap(
        raw,
        numa_mode="numa_aware",
        out_png="plots/fig3b_torcpy_numa_heatmap.png",
        out_pdf="plots/fig3b_torcpy_numa_heatmap.pdf",
        title=(
            "torcpy: Execution Time Across MPI-Process / Thread Configurations (NUMA-Aware Mode)"
        ),
    )
    print("✓  fig3b saved")


# =============================================================
# 6. FIGURE 4 — Parallel Efficiency
# =============================================================

def plot_fig4(best: pd.DataFrame):
    """
    1 γραμμή × 3 στήλες (μία ανά εφαρμογή).
    Y-άξονας: efficiency (%) = (speedup / total_cores) × 100, linear scale.
    Διακεκομμένη γραμμή στο 100% = ιδανική αξιοποίηση.
    Standard mode μόνο.
    """
    std = best[best["numa_mode"] == "standard"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for col, app in enumerate(APP_ORDER):
        ax = axes[col]

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

        # Ιδανική γραμμή 100%
        ax.axhline(100, linestyle=":", color="gray", linewidth=1.2)

        ax.set_xscale("log", base=2)
        ax.set_xticks(CORE_TICKS)
        ax.set_xticklabels(CORE_TICKS)
        ax.set_ylim(0, 115)   # 115 ώστε η γραμμή 100% να φαίνεται εντός plot
        ax.set_title(APP_TITLES[app])
        ax.set_xlabel("Total cores (best config)")
        if col == 0:
            ax.set_ylabel("Parallel efficiency (%)")

    axes[0].legend(loc="lower left", framealpha=0.9)

    fig.suptitle(
        "Parallel Efficiency: torcpy vs StarPUPy (Standard Mode, athena3)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig("plots/fig4_efficiency_comparison.png")
    fig.savefig("plots/fig4_efficiency_comparison.pdf")
    plt.close(fig)
    print("✓  fig4 saved")


# =============================================================
# 7. MAIN — τρέχει όλα τα γραφήματα με τη σειρά
# =============================================================

if __name__ == "__main__":
    set_style()

    print("Loading data...")
    raw  = load_raw()
    best = best_per_core(raw)
    print(f"  Raw rows   : {len(raw)}")
    print(f"  Best rows  : {len(best)}")
    print()

    print("Generating figures...")
    plot_fig1(best)
    plot_fig2(best)
    plot_fig3(raw)
    plot_fig3b(raw)
    plot_fig4(best)

    print()
    print("All figures saved in plots/")
    print("  PNG (preview)  → plots/fig*.png")
    print("  PDF (LaTeX)    → plots/fig*.pdf")
