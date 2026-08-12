#!/usr/bin/env python
"""
Scenario 1 operational comparison for the CEV paper.

Builds a 3-row x 3-column panel figure comparing the proposed strategy against
the two baselines in Scenario 1 (1 grid node, 1 MCS, 1 CEV):

  columns : Proposed | B1 | B2
  row (a) : MCS charge (grid->MCS, +) / discharge (MCS->CEV, -) [kW]
  row (b) : CEV state of energy [kWh] (with min/max limits) -> service quality
  row (c) : CEV total work power [kW]                       -> task completed

Result folders are named after the paper's nomenclature: B1_* and B2_* are the
paper's Baselines 1 and 2, and Reference_* is the rule-based reference scheduler
run that generates the frozen CEV schedule (not compared directly). Reads from
`results` (not results_old).

For a single MCS the grid charging power equals the MCS charge-from-grid power,
so the peak-shaving story lives entirely in row (a): the charge peak per strategy
is annotated. The 4-9 pm on-peak demand-charge window is shaded. Each row shares
its y-axis across all columns so the charge peaks are shown honestly on one scale.

Reads ONLY the Scenario 1 result CSVs; writes the figure into the paper's Figures
folder. Nothing inside the Scenario folders is modified.

Requires numpy + matplotlib (pip install -r requirements.txt).
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
import os

# Third-party dependencies (see requirements.txt in this folder).
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
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
RESULTS = os.path.join(HERE, "Scenario 1", "simple_dataset", "results")
# Paper figures directory (output target).
FIG_DIR = os.path.normpath(os.path.join(
    HERE))

# Run folders, named after the paper's nomenclature (B1/B2 = paper Baselines 1/2;
# the Reference_* run only generates the frozen CEV schedule and is not compared).
RUNS = {
    "Proposed": "OPTIMAL_Site_1_MCS_1_CEV_1_20260723_052306",
    "Baseline 1": "B1_Site_1_MCS_1_CEV_1_20260723_051947",
    "Baseline 2": "B2_Site_1_MCS_1_CEV_1_20260723_052126",
}

# --- Figure sizing tuned for the paper -------------------------------------
# In the paper this PNG sits in a 0.77*\textwidth minipage (IEEE journal
# \textwidth = 516 pt). We save at EXACTLY FIG_W_IN x FIG_H_IN (no tight bbox,
# so the on-page scale is known and fixed) and size every text element to a
# target *on-page* point size via onpage(). The (a)/(b)/(c) markers are set to
# 8 pt to exactly match the LaTeX \footnotesize "(d)" marker at this scale.
TEXTWIDTH_PT = 516.0
PT_PER_IN = 72.27
DISPLAY_FRAC = 0.77
DISPLAY_W_IN = TEXTWIDTH_PT / PT_PER_IN * DISPLAY_FRAC   # ~5.50 in on the page
FIG_W_IN = 14.0  # same overall width + aspect ratio as the 5-column version
FIG_H_IN = 9.5
SCALE = DISPLAY_W_IN / FIG_W_IN                          # native pt -> on-page pt


def onpage(pt):
    """Return the matplotlib fontsize that renders at `pt` points on the page
    once the PNG is scaled to its 0.77*\\textwidth width."""
    return pt / SCALE


PT_LETTER = 8.0   # (a)/(b)/(c) markers -- match the \footnotesize (d) marker
PT_LABEL = 6.0    # axis labels
PT_TICK = 5.0     # tick labels
PT_TITLE = 6.5    # column titles (Proposed/B1/B2)
PT_LEGEND = 6.5   # shared legend
PT_ANNOT = 5.0    # NCDP/OPDP annotations

DT_H = 0.25          # 15-min steps
PEAK_START_H = 16.0  # 4 pm  (on-peak window start, hours after 08:00 -> 8.0)
PEAK_END_H = 21.0    # 9 pm


def read_col(run, fname, col):
    """Return a float column from a result CSV as a numpy array."""
    path = os.path.join(RESULTS, RUNS[run], fname)
    vals = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                vals.append(float(row[col]))
            except (ValueError, KeyError):
                vals.append(np.nan)
    return np.asarray(vals)


def time_axis(n):
    """Hours-since-08:00 axis for n intervals (step starts)."""
    return 8.0 + np.arange(n) * DT_H


def clean(a):
    """Zero-out the tiny solver residuals (e.g. 3.6e-14) for clean plots."""
    a = np.asarray(a, dtype=float)
    a[np.abs(a) < 1e-6] = 0.0
    return a


def load(run):
    work = clean(read_col(run, "02_work_profiles_by_site.csv", "Total_Work_Power_kW"))
    mcs_ch = clean(read_col(run, "mcs_1_power_profile.csv", "Charging_Power_kW"))
    mcs_dch = clean(read_col(run, "mcs_1_power_profile.csv", "Discharging_Power_kW"))
    cev_soe = read_col(run, "04_cev_state_of_energy.csv", "CEV_1_SOE_kWh")
    soe_min = float(read_col(run, "04_cev_state_of_energy.csv", "CEV_1_Min_SOE_kWh")[0])
    soe_max = float(read_col(run, "04_cev_state_of_energy.csv", "CEV_1_Max_SOE_kWh")[0])
    return work, mcs_ch, mcs_dch, cev_soe, soe_min, soe_max


def hour_ticks(ax, t, labels=True, xmax=None, step_h=4):
    """Set an hourly x-tick grid; show rotated clock labels only if labels.

    xmax (hours-since-midnight) truncates the axis (used to zoom row (a) to the
    working portion of the day); step_h sets the tick spacing.
    """
    right = (xmax if xmax is not None else t[-1] + DT_H)
    ticks = np.arange(8, right + 0.1, step_h)
    ax.set_xticks(ticks)
    if labels:
        ax.set_xticklabels([f"{int(h) % 24:02d}:00" for h in ticks],
                           fontsize=onpage(PT_TICK), rotation=45, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlim(t[0], right)


def step(ax, t, y, **kw):
    """Step plot aligned to interval starts."""
    ax.step(np.append(t, t[-1] + DT_H), np.append(y, y[-1]),
            where="post", **kw)


# CEV work subactivities, keyed by their characteristic power [kW] (parameters.csv).
SUBACTS = [
    ("Digging", 4.79, "#4c72b0"),
    ("Traveling", 4.71, "#dd8452"),
    ("Loading+swinging", 3.16, "#55a868"),
]


def classify_subact(work):
    """Map each interval's work power to the nearest subactivity index, or -1 (idle)."""
    powers = np.array([p for _, p, _ in SUBACTS])
    cls = np.full(len(work), -1, dtype=int)
    for i, w in enumerate(work):
        if w > 0.5:
            cls[i] = int(np.argmin(np.abs(powers - w)))
    return cls


C_SOE = "black"      # CEV state of energy
C_CH = "#d95f02"     # MCS charging from grid
C_DCH = "#e7298a"    # MCS discharging to CEV
C_OPDP = "#b2182b"   # on-peak demand peak annotation
PEAK_SHADE = dict(color="0.85", alpha=0.6, zorder=0)


# Row order: (a) MCS charge/discharge, (b) CEV state of energy, (c) CEV work.
R_MCS, R_SOE, R_WORK = 0, 1, 2


def make_figure(cols, data, ncdp, opdp, work_max, mcs_lo, mcs_hi,
                soe_min, soe_max, soe_pad, t, out_stub, zoom_work_end=None,
                fig_h=FIG_H_IN):
    """Build+save the 3x3 operation figure.

    zoom_work_end: if given (hours-since-midnight, e.g. 18.0), the CEV-work row
    (c) is truncated to [08:00, zoom_work_end] so the subactivities read more
    clearly. Because the bottom row's axis then differs from the others, rows
    (a) and (b) each get their own clock ticks + "Time of day" label too, so
    every panel is self-contained.
    """
    own_axis = zoom_work_end is not None  # every row carries its own x furniture
    fig, axes = plt.subplots(3, len(cols), figsize=(FIG_W_IN, fig_h),
                             sharex=False)

    for j, col in enumerate(cols):
        work, mcs_ch, mcs_dch, cev_soe, smin, smax = data[col]
        t_soe = time_axis(len(cev_soe))  # SOE is a state -> one more point

        for i in range(3):
            axes[i, j].axvspan(PEAK_START_H, PEAK_END_H, **PEAK_SHADE)
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].tick_params(labelsize=onpage(PT_TICK))

        # row (a): MCS charge (+) from grid, discharge (-) to CEV
        step(axes[R_MCS, j], t, mcs_ch, color=C_CH, lw=1.8, label="MCS charge")
        step(axes[R_MCS, j], t, -mcs_dch, color=C_DCH, lw=1.8,
             label="MCS discharge")
        axes[R_MCS, j].axhline(0, color="0.5", lw=1.0)
        axes[R_MCS, j].set_ylim(mcs_lo, mcs_hi)
        # NCDP across the full horizon; OPDP only within the on-peak window
        axes[R_MCS, j].axhline(ncdp[col], color=C_CH, lw=1.2, ls=":")
        axes[R_MCS, j].plot([PEAK_START_H, PEAK_END_H], [opdp[col], opdp[col]],
                            color=C_OPDP, lw=1.8, ls=":")
        # NCDP/OPDP: label + value together, stacked in the bottom-right corner
        # so they stay clear of the (central) charge spike.
        axes[R_MCS, j].annotate(f"NCDP {ncdp[col]:.2f} kW", xy=(0.97, 0.15),
                                xycoords="axes fraction", ha="right",
                                va="bottom", fontsize=onpage(PT_ANNOT),
                                color=C_CH, fontweight="bold")
        axes[R_MCS, j].annotate(f"OPDP {opdp[col]:.2f} kW", xy=(0.97, 0.03),
                                xycoords="axes fraction", ha="right",
                                va="bottom", fontsize=onpage(PT_ANNOT),
                                color=C_OPDP, fontweight="bold")

        # row (b): CEV state of energy with min/max bounds
        axes[R_SOE, j].plot(t_soe, cev_soe, color=C_SOE, lw=2.0)
        axes[R_SOE, j].axhline(smin, color="0.5", lw=1.0, ls="--")
        axes[R_SOE, j].axhline(smax, color="0.5", lw=1.0, ls="--")
        axes[R_SOE, j].set_ylim(soe_min - soe_pad, soe_max + soe_pad)

        # row (c): CEV work power, filled per subactivity
        te = np.append(t, t[-1] + DT_H)
        cls = classify_subact(work)
        for k, (name, _pwr, color) in enumerate(SUBACTS):
            ys = np.where(cls == k, work, 0.0)
            axes[R_WORK, j].fill_between(te, np.append(ys, ys[-1]), step="post",
                                         color=color, alpha=0.85, lw=0,
                                         label=(name if j == 0 else None))
        step(axes[R_WORK, j], t, work, color="0.3", lw=1.0)  # crisp outline
        axes[R_WORK, j].set_ylim(0, work_max)

        axes[0, j].set_title(col, fontsize=onpage(PT_TITLE), fontweight="bold")
        # x-tick grid on every row. Clock labels + "Time of day" normally go only
        # on the bottom row (all rows share one time axis). When the work row is
        # zoomed its axis differs, so every row gets its own labels.
        if own_axis:
            hour_ticks(axes[0, j], t, labels=True)
            axes[0, j].set_xlabel("Time of day", fontsize=onpage(PT_LABEL))
            hour_ticks(axes[1, j], t, labels=True)
            axes[1, j].set_xlabel("Time of day", fontsize=onpage(PT_LABEL))
            hour_ticks(axes[R_WORK, j], t, labels=True, xmax=zoom_work_end,
                       step_h=2)
        else:
            hour_ticks(axes[0, j], t, labels=False)
            hour_ticks(axes[1, j], t, labels=False)
            hour_ticks(axes[2, j], t, labels=True)
        axes[2, j].set_xlabel("Time of day", fontsize=onpage(PT_LABEL))

    # y-labels only on the left column
    axes[R_MCS, 0].set_ylabel("MCS power\n[kW]", fontsize=onpage(PT_LABEL))
    axes[R_SOE, 0].set_ylabel("CEV state of\nenergy [kWh]",
                              fontsize=onpage(PT_LABEL))
    axes[R_WORK, 0].set_ylabel("CEV work\npower [kW]", fontsize=onpage(PT_LABEL))

    # legend handles, in panel order: charge/discharge (row a) + NCDP/OPDP +
    # subactivities (row c) + SOE limits + on-peak band
    h0, l0 = axes[R_MCS, 0].get_legend_handles_labels()    # charge / discharge
    h2, l2 = axes[R_WORK, 0].get_legend_handles_labels()   # subactivity fills
    ncdp_proxy = matplotlib.lines.Line2D([], [], color=C_CH, lw=0.9, ls=":",
                                         label="NCDP")
    opdp_proxy = matplotlib.lines.Line2D([], [], color=C_OPDP, lw=1.4, ls=":",
                                         label="OPDP")
    bound = matplotlib.lines.Line2D([], [], color="0.5", lw=0.8, ls="--",
                                    label="CEV SOE limits")
    band = matplotlib.patches.Patch(**{k: v for k, v in PEAK_SHADE.items()
                                        if k in ("color", "alpha")},
                                    label="On-peak (4--9 pm)")
    # strict panel order: (a) MCS + peaks, (b) SOE limits, (c) subactivities, band
    handles = h0 + [ncdp_proxy, opdp_proxy, bound] + h2 + [band]
    labels = l0 + ["NCDP", "OPDP", "CEV SOE limits"] + l2 + ["On-peak (4--9 pm)"]

    # When every row carries its own ticks + x-label they each need room below
    # them -> a larger inter-row gap, and each panel letter sits just below its
    # own x-label (uniform offset since every row then has furniture).
    hspace = 0.74 if own_axis else 0.32
    if own_axis:
        letter_off = [0.082, 0.082, 0.082]
        legend_y = 0.030
    else:
        letter_off = [0.015, 0.015, 0.098]
        legend_y = 0.004

    # Tightened layout: less inter-row whitespace (bigger panels), a small
    # reserved bottom band for the shared legend. We do NOT crop with a tight
    # bbox on save, so the panels fill the canvas and the on-page scale is exact.
    fig.tight_layout(rect=[0.0, 0.13, 1.0, 0.99])
    fig.subplots_adjust(hspace=hspace)

    # panel labels (a)/(b)/(c) centred below each row.
    for i, letter in enumerate(["(a)", "(b)", "(c)"]):
        y0 = axes[i, 0].get_position().y0
        fig.text(0.5, y0 - letter_off[i], letter, ha="center", va="top",
                 fontsize=onpage(PT_LETTER))

    # shared legend flush at the bottom of the reserved band
    fig.legend(handles, labels, ncol=5, fontsize=onpage(PT_LEGEND),
               loc="lower center", bbox_to_anchor=(0.5, legend_y),
               columnspacing=1.3, handlelength=1.6, handletextpad=0.5,
               framealpha=0.9)

    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, out_stub + ".png")
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("Saved:", png)


def main():
    cols = ["Proposed", "Baseline 1", "Baseline 2"]
    data = {r: load(r) for r in cols}

    n = len(data["Proposed"][0])
    t = time_axis(n)

    # Demand peaks per strategy: NCDP = max grid draw over all intervals;
    # OPDP = max grid draw within the on-peak window (these set NCDC/OPDC).
    onpeak = (t >= PEAK_START_H) & (t < PEAK_END_H)
    ncdp = {r: float(np.max(data[r][1])) for r in cols}
    opdp = {r: (float(np.max(data[r][1][onpeak])) if onpeak.any() else 0.0)
            for r in cols}

    # shared y-limits per row so columns are comparable
    work_max = max(np.max(data[c][0]) for c in cols) * 1.2
    mcs_vals = np.concatenate([np.r_[data[c][1], -data[c][2]] for c in cols])
    mcs_lo, mcs_hi = mcs_vals.min() * 1.15, mcs_vals.max() * 1.12
    soe_min = data["Proposed"][4]
    soe_max = data["Proposed"][5]
    soe_pad = 0.12 * (soe_max - soe_min)

    shared = (cols, data, ncdp, opdp, work_max, mcs_lo, mcs_hi,
              soe_min, soe_max, soe_pad, t)

    # Zoomed figure (the paper's version): the CEV-work row (c) is zoomed to
    # 08:00-18:00 and every row gets its own x-axis; taller canvas for the
    # extra labels.
    make_figure(*shared, out_stub="2_scenario1_operation_zoom",
                zoom_work_end=18.0, fig_h=11.5)

    print("\nDemand peaks per strategy [kW]:")
    print(f"  {'':11s} {'NCDP':>7s} {'OPDP':>7s}")
    for r in cols:
        print(f"  {r:11s} {ncdp[r]:7.2f} {opdp[r]:7.2f}")


if __name__ == "__main__":
    import matplotlib.patches  # noqa: E402  (used in main)
    import matplotlib.lines    # noqa: E402  (used in main)
    main()
