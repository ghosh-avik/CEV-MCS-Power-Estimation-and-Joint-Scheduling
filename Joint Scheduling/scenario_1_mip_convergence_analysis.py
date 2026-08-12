#!/usr/bin/env python
# we are great
"""
MIP convergence analysis for the CEV paper — Scenario 4, proposed (OPTIMAL) run.

Goal
----
Scenario 4 is the only case whose proposed run does not prove optimality within
the one-hour limit (Status == TIME_LIMIT, final MIP gap ~11.9 %). This script
reads that single run's solver trajectory from `10_mip_convergence.csv` and draws
ONE plot of the convergence versus elapsed wall-clock time:
  * left  y-axis: MIP gap [%]                        (gap_pct)
  * right y-axis: normalized primal solution         (best_sol / max|best_sol|)

Notes
-----
* `best_sol` is the incumbent (best feasible cost found so far), reported as `Inf`
  until the first incumbent appears; those rows are dropped. The primal objective
  is normalised by the maximum absolute incumbent so it lands in [0, 1] and decays.
* This script ONLY READS from Scenario 4/simple_dataset/results (NOT results_old)
  and writes its outputs (figure + summary CSV) into this "Paper Scenarios"
  directory. It never writes inside the Scenario folders.
"""

"""""""""""""""""""""""""""""""""""""""IMPORT PACKAGES HERE"""""""""""""""""""""""""""""""""

import os, sys

# Clear any variables from a previous run (only meaningful in Spyder / IPython /
# Jupyter, where the kernel persists; a plain `python file.py` already starts fresh).
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("reset", "-f")
except ModuleNotFoundError:
    pass

# Clear terminal (visible screen + scrollback). Works in VS Code's integrated
# terminal, macOS Terminal, iTerm2, and Linux. `os.system("clear")` only clears
# the visible portion in some terminals (notably VS Code), leaving scrollback.
if os.name == "nt":
    os.system("cls")
else:
    print("\033[H\033[2J\033[3J", end="", flush=True)

import csv
import glob
import os

# Third-party dependencies (see requirements.txt in this folder).
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend, no display needed
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    sys.exit(
        f"\nMissing dependency: {e.name!r}.\n\n"
        "Install the requirements into your Python environment first:\n\n"
        "    pip install -r requirements.txt\n\n"
        "(run from this script's folder, inside the environment you use to run it)"
    )

HERE = os.path.dirname(os.path.abspath(__file__))

# The runs we analyse: Scenario 4 proposed solve under a 1-hour limit (OPTIMAL_*)
# and the extended 5-hour re-solve (Extended_OPTIMAL_*), both in `results`.
SCENARIO_DIR = os.path.join(HERE, "Scenario 4", "simple_dataset", "results")

# --- Figure sizing tuned for the paper -------------------------------------
# In the paper this PNG is included at \columnwidth (IEEE journal column =
# 252 pt). The figure is FIG_W_IN wide, so text set via onpage() renders close
# to the target on-page point size (same scheme/targets as the scenario plot).
COLUMNWIDTH_PT = 252.0
PT_PER_IN = 72.27
DISPLAY_W_IN = COLUMNWIDTH_PT / PT_PER_IN               # ~3.49 in on the page
FIG_W_IN = 8.0
FIG_H_IN = 5.0
SCALE = DISPLAY_W_IN / FIG_W_IN                         # native pt -> on-page pt


def onpage(pt):
    """Matplotlib fontsize that renders at ~`pt` points at \\columnwidth."""
    return pt / SCALE


PT_LABEL = 6.5    # axis labels
PT_TICK = 5.5     # x/y tick labels
PT_LEGEND = 6.5   # legend

# Colour encodes the RUN (1 h vs 5 h limit); line style encodes the QUANTITY
# (solid = MIP gap on the left axis, dashed = normalized incumbent on the right).
C_1H = "#d95f02"   # 1-hour run
C_5H = "#1f78b4"   # 5-hour (extended) run


def read_kpi(run_dir):
    """Read 09_cost_kpi_metrics.csv into a {metric: value} dict (strings)."""
    path = os.path.join(run_dir, "09_cost_kpi_metrics.csv")
    out = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) == 2:
                out[row[0].strip()] = row[1].strip()
    return out


def read_convergence(run_dir):
    """Return (time_s, best_sol, gap_pct) arrays with Inf/NaN rows removed.

    Rows before the first incumbent have best_sol = gap_pct = Inf and are
    dropped, so all three arrays are aligned and finite.
    """
    path = os.path.join(run_dir, "10_mip_convergence.csv")
    times, sols, gaps = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                t = float(row["time_s"])
                s = float(row["best_sol"])  # 'Inf' -> inf
                g = float(row["gap_pct"])   # 'Inf' -> inf
            except (ValueError, KeyError):
                continue
            if np.isfinite(t) and np.isfinite(s) and np.isfinite(g):
                times.append(t)
                sols.append(s)
                gaps.append(g)
    return np.asarray(times), np.asarray(sols), np.asarray(gaps)


def find_run(prefix):
    """Locate a Scenario 4 run folder by name prefix (most recent if several)."""
    matches = sorted(
        d for d in glob.glob(os.path.join(SCENARIO_DIR, prefix + "*"))
        if os.path.isdir(d)
    )
    if not matches:
        raise SystemExit(f"No {prefix}* run found in {SCENARIO_DIR}")
    return matches[-1]


ONE_HOUR_S = 3600.0  # the 1-hour limit; a marker line separates the two runs


def main():
    # Two runs to overlay: the 1-hour limited solve and the 5-hour extended one.
    specs = [
        ("1 h limit", "OPTIMAL_", C_1H),
        ("5 h limit", "Extended_OPTIMAL_", C_5H),
    ]

    runs = []
    for label, prefix, color in specs:
        run_dir = find_run(prefix)
        kpi = read_kpi(run_dir)
        final_gap = float(kpi.get("MIP_Gap_percent", "nan"))
        status = kpi.get("Status", "?")
        t, s, gtraj = read_convergence(run_dir)
        if t.size == 0:
            raise SystemExit(f"No finite incumbents in {run_dir}")
        runs.append(dict(label=label, color=color, t=t, s=s, gtraj=gtraj,
                         final_gap=final_gap, status=status,
                         name=os.path.basename(run_dir)))
        print(f"{label}: {os.path.basename(run_dir)}  status={status}  "
              f"final gap={final_gap:.2f}%  t_end={t[-1]:.0f}s")

    # Normalise both incumbents by the SAME scale so the two are comparable.
    global_max_abs = max(np.max(np.abs(r["s"])) for r in runs)
    for r in runs:
        r["norm"] = r["s"] / global_max_abs
    x_max = max(r["t"][-1] for r in runs)

    # ---- Plot: one figure, twin y-axes ------------------------------------
    fig, ax_gap = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax_sol = ax_gap.twinx()

    gap_handles, sol_handles = [], []
    for r in runs:
        t, gtraj, norm, color = r["t"], r["gtraj"], r["norm"], r["color"]
        me = max(1, t.size // 14)  # ~14 markers so the line stays readable
        lg, = ax_gap.plot(t, gtraj, color=color, linewidth=2.0, linestyle="-",
                          marker="o", markersize=6, markevery=me,
                          label=f"MIP gap ({r['label']})")
        ls_, = ax_sol.plot(t, norm, color=color, linewidth=2.0, linestyle="--",
                           marker="s", markersize=6, markevery=me,
                           label=f"Incumbent ({r['label']})")
        gap_handles.append(lg)
        sol_handles.append(ls_)

    # Vertical dotted line at the 1-hour mark separating the two runs.
    ax_gap.axvline(ONE_HOUR_S, color="0.35", linestyle=":", linewidth=1.5,
                   zorder=1)
    ax_gap.annotate("1 h limit", xy=(ONE_HOUR_S, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -3), textcoords="offset points",
                    ha="left", va="top", color="0.35",
                    fontsize=onpage(PT_TICK))

    ax_gap.set_xlabel("Elapsed solve time [s]", fontsize=onpage(PT_LABEL))
    ax_gap.set_ylabel("MIP gap [%]", fontsize=onpage(PT_LABEL))
    ax_sol.set_ylabel("Normalized incumbent objective", fontsize=onpage(PT_LABEL))

    ax_gap.tick_params(axis="both", labelsize=onpage(PT_TICK))
    ax_sol.tick_params(axis="y", labelsize=onpage(PT_TICK))

    ax_gap.set_xlim(left=0.0, right=x_max * 1.01)
    ax_gap.set_xticks(np.arange(0, x_max + 1, ONE_HOUR_S))
    ax_gap.set_ylim(bottom=0.0)
    ax_sol.set_ylim(0.0, 1.05)
    ax_gap.grid(True, alpha=0.3)

    # Legend: solid = MIP gap (left axis), dashed = incumbent (right axis);
    # colour distinguishes the 1 h and 5 h runs.
    handles = gap_handles + sol_handles
    ax_gap.legend(handles=handles, loc="upper right",
                  fontsize=onpage(PT_LEGEND), framealpha=0.9,
                  handlelength=2.2, labelspacing=0.4)

    fig.tight_layout()

    png = os.path.join(HERE, "2_mip_convergence_normalized.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {png}")

    # ---- Summary CSV ------------------------------------------------------
    out_csv = os.path.join(HERE, "2_mip_convergence_summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "n_points", "t_first_incumbent_s", "t_last_s",
                    "first_best_sol", "final_best_sol", "max_abs_best_sol",
                    "final_gap_pct"])
        for r in runs:
            t, s = r["t"], r["s"]
            w.writerow([r["label"], t.size, f"{t[0]:.2f}", f"{t[-1]:.2f}",
                        f"{s[0]:.3f}", f"{s[-1]:.3f}", f"{global_max_abs:.3f}",
                        f"{r['final_gap']:.2f}"])
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
