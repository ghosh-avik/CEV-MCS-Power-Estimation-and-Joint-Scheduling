#!/usr/bin/env python
"""
Scenario 2 operational comparison for the CEV paper.

Builds a 3-row x 3-column panel figure comparing the proposed strategy against
the two baselines in Scenario 2 (1 grid node, 1 MCS, 2 CEVs):

  columns : Proposed | Baseline 1 | Baseline 2
  row (a) : MCS charge (grid->MCS, +) / discharge (MCS->CEV, -) [kW]
  row (b) : CEV state of energy [kWh] -- CEV 1 and CEV 2 in different colors
  row (c) : CEV work power [kW] -- STACKED bars for CEV 1 + CEV 2

Per-CEV work power is not exported directly (02_work_profiles_by_site.csv only
stores the site total), so it is reconstructed from the solver's activity
indicator u[e,i,a,k] in 11_cev_schedule_u_mu.csv:
    P_work[e,k] = sum_a p_activity[a] * u[e,i,a,k]
with the activity powers from parameters.csv (p_digging=4.77, p_loading_swinging
=3.18, p_traveling=4.72 kW). Summed over CEVs this matches Total_Work_Power_kW.

Result folders are named after the paper's nomenclature: B1_* and B2_* are the
paper's Baselines 1 and 2, and Reference_* is the rule-based reference scheduler
run that generates the frozen CEV schedule (not compared directly). Reads from
`results` (not results_old). Writes figures into this directory under NEW names
so the Scenario 1 figures are not overwritten. Nothing inside the Scenario
folders is modified.

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

# Clear terminal (visible screen + scrollback).
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
RESULTS = os.path.join(HERE, "Scenario 2", "simple_dataset", "results")
# Output target (this directory).
FIG_DIR = os.path.normpath(os.path.join(HERE))

# Run folders, named after the paper's nomenclature (B1/B2 = paper Baselines 1/2;
# the Reference_* run only generates the frozen CEV schedule and is not compared).
RUNS = {
    "Proposed": "OPTIMAL_Site_1_MCS_1_CEV_2_20260723_174926",
    "Baseline 1": "B1_Site_1_MCS_1_CEV_2_20260723_174715",
    "Baseline 2": "B2_Site_1_MCS_1_CEV_2_20260723_174815",
}

# CEVs present in this scenario.
E = [1, 2]

# Activity index -> nominal power [kW] (from parameters.csv, matching the model's
# p_activity: a=1 digging, a=2 loading/swinging, a=3 traveling).
P_ACT = {1: 4.77, 2: 3.18, 3: 4.72}

# --- Figure sizing tuned for the paper -------------------------------------
# In the paper this PNG sits in a 0.77*\textwidth minipage (IEEE journal
# \textwidth = 516 pt). We save at EXACTLY FIG_W_IN x FIG_H_IN (no tight bbox,
# so the on-page scale is known and fixed) and size every text element to a
# target *on-page* point size via onpage().
TEXTWIDTH_PT = 516.0
PT_PER_IN = 72.27
DISPLAY_FRAC = 0.77
DISPLAY_W_IN = TEXTWIDTH_PT / PT_PER_IN * DISPLAY_FRAC   # ~5.50 in on the page
FIG_W_IN = 14.0  # same overall width + aspect ratio as the Scenario 1 figure
FIG_H_IN = 9.5
SCALE = DISPLAY_W_IN / FIG_W_IN                          # native pt -> on-page pt


def onpage(pt):
    """Return the matplotlib fontsize that renders at `pt` points on the page
    once the PNG is scaled to its 0.77*\\textwidth width."""
    return pt / SCALE


PT_LETTER = 8.0   # (a)/(b)/(c) markers
PT_LABEL = 6.0    # axis labels
PT_TICK = 5.0     # tick labels
PT_TITLE = 6.5    # column titles (Proposed/Baseline 1/Baseline 2)
PT_LEGEND = 6.5   # shared legend
PT_ANNOT = 5.0    # NCDP/OPDP annotations

DT_H = 0.25          # 15-min steps
PEAK_START_H = 16.0  # 4 pm  (on-peak window start, hours after 08:00 -> 8.0)
PEAK_END_H = 21.0    # 9 pm

# row (c): colour the stacked CEV work bars by subactivity (previous colour code,
# keyed by activity index a: 1 digging, 2 loading/swinging, 3 traveling).
SUBACTS = [
    ("Digging", 1, "#4c72b0"),
    ("Traveling", 3, "#dd8452"),
    ("Loading+swinging", 2, "#55a868"),
]
# row (b): CEV state-of-energy line colours (CEV 1 black, CEV 2 distinct colour).
C_SOE = {1: "black", 2: "#d62728"}
C_CH = "#d95f02"     # MCS charging from grid
C_DCH = "#e7298a"    # MCS discharging to CEV
C_OPDP = "#b2182b"   # on-peak demand peak annotation
PEAK_SHADE = dict(color="0.85", alpha=0.6, zorder=0)


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


def read_cev_work(run, n):
    """Reconstruct per-CEV work power + activity on the K grid from u[e,i,a,k].

    Returns (pow, act) where pow[e][k] = sum_a p_activity[a] * u[e,i,a,k] and
    act[e][k] = the activity index performed (0 = idle). A CEV performs at most
    one activity per interval, so act is well defined.
    """
    path = os.path.join(RESULTS, RUNS[run], "11_cev_schedule_u_mu.csv")
    pow_ = {e: np.zeros(n) for e in E}
    act = {e: np.zeros(n, dtype=int) for e in E}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0] != "u":
                continue
            try:
                e, _i, a, k = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                val = float(row[5])
            except (ValueError, IndexError):
                continue
            if e in pow_ and 1 <= k <= n and a in P_ACT and val > 0.5:
                pow_[e][k - 1] += P_ACT[a]
                act[e][k - 1] = a
    return pow_, act


def time_axis(n):
    """Hours-since-08:00 axis for n intervals (step starts)."""
    return 8.0 + np.arange(n) * DT_H


def clean(a):
    """Zero-out the tiny solver residuals (e.g. 3.6e-14) for clean plots."""
    a = np.asarray(a, dtype=float)
    a[np.abs(a) < 1e-6] = 0.0
    return a


def load(run):
    mcs_ch = clean(read_col(run, "mcs_1_power_profile.csv", "Charging_Power_kW"))
    mcs_dch = clean(read_col(run, "mcs_1_power_profile.csv", "Discharging_Power_kW"))
    n = len(mcs_ch)
    cev_pow, cev_act = read_cev_work(run, n)  # {e: array(n)}
    cev_soe = {e: read_col(run, "04_cev_state_of_energy.csv", f"CEV_{e}_SOE_kWh")
               for e in E}
    lims = {e: (float(read_col(run, "04_cev_state_of_energy.csv",
                               f"CEV_{e}_Min_SOE_kWh")[0]),
                float(read_col(run, "04_cev_state_of_energy.csv",
                               f"CEV_{e}_Max_SOE_kWh")[0]))
            for e in E}
    return cev_pow, cev_act, mcs_ch, mcs_dch, cev_soe, lims


def hour_ticks(ax, t, labels=True, xmax=None, step_h=4):
    """Set an hourly x-tick grid; show rotated clock labels only if labels.

    xmax (hours-since-midnight) truncates the axis (used to zoom the work row to the
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


def stepfill(ax, t, lo, hi, **kw):
    """Filled step band between lo and hi, aligned to interval starts."""
    ax.fill_between(np.append(t, t[-1] + DT_H),
                    np.append(lo, lo[-1]), np.append(hi, hi[-1]),
                    step="post", **kw)


# Row order: (a) MCS charge/discharge, (b) CEV state of energy, (c) CEV work.
R_MCS, R_SOE, R_WORK = 0, 1, 2


def make_figure(cols, data, ncdp, opdp, work_max, mcs_lo, mcs_hi,
                soe_lo, soe_hi, t, out_stub, zoom_work_end=None,
                fig_h=FIG_H_IN):
    """Build+save the 3x3 operation figure.

    zoom_work_end: if given (hours-since-midnight, e.g. 18.0), the CEV-work row
    (c) is truncated to [08:00, zoom_work_end] so the subactivities read more
    clearly. Because the bottom row's axis then differs from the others, rows
    (a) and (b) each get their own clock ticks + "Time of day" label too, so
    every panel is self-contained.
    """
    own_axis = zoom_work_end is not None
    fig, axes = plt.subplots(3, len(cols), figsize=(FIG_W_IN, fig_h),
                             sharex=False)

    for j, col in enumerate(cols):
        cev_pow, cev_act, mcs_ch, mcs_dch, cev_soe, lims = data[col]
        n = len(mcs_ch)
        t_soe = time_axis(len(cev_soe[E[0]]))  # SOE is a state -> one more point

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
        # NCDP/OPDP: label + value together, stacked in the bottom-right corner.
        axes[R_MCS, j].annotate(f"NCDP {ncdp[col]:.2f} kW", xy=(0.97, 0.15),
                                xycoords="axes fraction", ha="right",
                                va="bottom", fontsize=onpage(PT_ANNOT),
                                color=C_CH, fontweight="bold")
        axes[R_MCS, j].annotate(f"OPDP {opdp[col]:.2f} kW", xy=(0.97, 0.03),
                                xycoords="axes fraction", ha="right",
                                va="bottom", fontsize=onpage(PT_ANNOT),
                                color=C_OPDP, fontweight="bold")

        # row (b): CEV state of energy, one line per CEV + shared min/max bounds
        for e in E:
            axes[R_SOE, j].plot(t_soe, cev_soe[e], color=C_SOE[e], lw=2.0,
                                label=(f"CEV {e} SOE" if j == 0 else None))
        for lo, hi in {lims[e] for e in E}:
            axes[R_SOE, j].axhline(lo, color="0.5", lw=1.0, ls="--")
            axes[R_SOE, j].axhline(hi, color="0.5", lw=1.0, ls="--")
        axes[R_SOE, j].set_ylim(soe_lo, soe_hi)

        # row (c): CEV work power, STACKED over CEVs (CEV 1 bottom, CEV 2 on top),
        # each CEV's band coloured by the subactivity it performs.
        base = np.zeros(n)
        for e in E:
            for name, a_idx, color in SUBACTS:
                seg = np.where(cev_act[e] == a_idx, cev_pow[e], 0.0)
                stepfill(axes[R_WORK, j], t, base, base + seg, color=color,
                         alpha=0.85, lw=0,
                         label=(name if (j == 0 and e == E[0]) else None))
            base = base + cev_pow[e]
        step(axes[R_WORK, j], t, base, color="0.3", lw=1.0)  # total outline
        axes[R_WORK, j].set_ylim(0, work_max)

        axes[0, j].set_title(col, fontsize=onpage(PT_TITLE), fontweight="bold")
        # x furniture. Normally only the bottom row carries clock labels; when the
        # work row is zoomed its axis differs, so every row gets its own labels.
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

    # legend in panel order: MCS charge/discharge + NCDP/OPDP (row a) + CEV SOE
    # lines + limits (row b) + subactivity fills (row c) + on-peak band
    h0, l0 = axes[R_MCS, 0].get_legend_handles_labels()    # charge / discharge
    h1, l1 = axes[R_SOE, 0].get_legend_handles_labels()    # CEV 1 / CEV 2 SOE
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
    handles = h0 + [ncdp_proxy, opdp_proxy] + h1 + [bound] + h2 + [band]
    labels = (l0 + ["NCDP", "OPDP"] + l1 + ["CEV SOE limits"] + l2
              + ["On-peak (4--9 pm)"])

    # When every row carries its own ticks + x-label they each need room below
    # them -> a larger inter-row gap, and each panel letter sits just below its
    # own x-label (uniform offset since every row then has furniture).
    hspace = 0.74 if own_axis else 0.32
    if own_axis:
        letter_off = [0.082, 0.082, 0.082]
        legend_y = 0.008
        rect_bottom = 0.15
    else:
        letter_off = [0.015, 0.015, 0.105]
        legend_y = 0.004
        rect_bottom = 0.24  # bigger reserve: (c) fits between x-label and legend

    # Reserve a bottom band for the shared legend. We do NOT crop with a tight
    # bbox on save, so the panels fill the canvas and the on-page scale is exact.
    fig.tight_layout(rect=[0.0, rect_bottom, 1.0, 0.99])
    fig.subplots_adjust(hspace=hspace)

    # panel labels (a)/(b)/(c) centred below each row.
    for i, letter in enumerate(["(a)", "(b)", "(c)"]):
        y0 = axes[i, 0].get_position().y0
        fig.text(0.5, y0 - letter_off[i], letter, ha="center", va="top",
                 fontsize=onpage(PT_LETTER))

    fig.legend(handles, labels, ncol=4, fontsize=onpage(PT_LEGEND),
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

    n = len(data["Proposed"][2])  # mcs_ch length
    t = time_axis(n)

    # Demand peaks per strategy: NCDP = max grid draw over all intervals;
    # OPDP = max grid draw within the on-peak window (these set NCDC/OPDC).
    onpeak = (t >= PEAK_START_H) & (t < PEAK_END_H)
    ncdp = {r: float(np.max(data[r][2])) for r in cols}
    opdp = {r: (float(np.max(data[r][2][onpeak])) if onpeak.any() else 0.0)
            for r in cols}

    # shared y-limits per row so columns are comparable
    work_max = max(np.max(sum(data[c][0][e] for e in E)) for c in cols) * 1.2
    mcs_vals = np.concatenate([np.r_[data[c][2], -data[c][3]] for c in cols])
    mcs_lo, mcs_hi = mcs_vals.min() * 1.15, mcs_vals.max() * 1.12
    all_lims = [v for c in cols for e in E for v in data[c][5][e]]
    lo, hi = min(all_lims), max(all_lims)
    pad = 0.12 * (hi - lo)
    soe_lo, soe_hi = lo - pad, hi + pad

    shared = (cols, data, ncdp, opdp, work_max, mcs_lo, mcs_hi, soe_lo, soe_hi, t)

    # Zoomed figure (the paper's version): the CEV-work row (c) is zoomed to
    # 08:00-18:00 and every row gets its own x-axis; taller canvas for the
    # extra labels.
    make_figure(*shared, out_stub="3_scenario2_operation_zoom",
                zoom_work_end=18.0, fig_h=11.5)

    print("\nDemand peaks per strategy [kW]:")
    print(f"  {'':11s} {'NCDP':>7s} {'OPDP':>7s}")
    for r in cols:
        print(f"  {r:11s} {ncdp[r]:7.2f} {opdp[r]:7.2f}")


if __name__ == "__main__":
    import matplotlib.patches  # noqa: E402  (used in main)
    import matplotlib.lines    # noqa: E402  (used in main)
    main()
