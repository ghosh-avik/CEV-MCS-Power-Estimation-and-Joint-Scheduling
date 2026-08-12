#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 01:23:12 2025

@author: avikghosh
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

# Third-party dependencies (see requirements.txt in this folder).
try:
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    import cvxpy as cp
    from scipy.optimize import nnls
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import RepeatedKFold
except ImportError as e:
    sys.exit(
        f"\nMissing dependency: {e.name!r}.\n\n"
        "Install the requirements into your Python environment first:\n\n"
        "    pip install -r requirements.txt\n\n"
        "(run from this script's folder, inside the environment you use to run it)"
    )

plt.ion()  # non-blocking plt.show(); script keeps running while figures stay open

import math
import time
from time import process_time
from datetime import datetime, timedelta
import re
import statistics


start_time = time.time()

"""""""""""""""""""""""""""""""""""""""READ INPUT DATA HERE"""""""""""""""""""""""""""""""""

# Task-recording Excel files live in the repo's Raw Data folder, next to this script.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Raw Data")

# Equation weighting scheme. See compute_weights() for the available options:
#   "uniform"        – every equation gets weight 1
#   "linear"         – w ∝ |b| (sharp)
#   "bounded_median" – w = min(|b|/median, 1)  (only sub-median ΔSoC equations are penalized)
#   "quadratic"      – w ∝ b² (very sharp; effectively excludes the smallest ΔSoC equations)
WEIGHT_SCHEME = "uniform"

# Minimum cumulative |ΔSoC| (in %) per equation. Activities accumulate into a
# bucket until cumulative SoC drop reaches this threshold, then one equation is
# emitted for the full bucket. Set to 1 to reproduce the original behavior of
# one equation per SoC step. Larger values trade equation count for per-equation
# signal-to-noise (1 % steps carry ~50 % relative quantization noise; 2 % steps
# carry ~25 %).
MIN_DELTA_SOC = 3

# Material handled. Selects which task-recording files feed the analysis, so you
# don't have to hand-edit all_task_dfs. One of: "soil", "decomposed granite",
# "sand", "all".
#
# NOTE: Data_tasks_1 .. Data_tasks_23 are 23 individual task-recording files
# (Excel sheets), NOT 23 days. They were collected over 9 calendar days
# (Oct 21-23 2025 and Feb 02-13 2026); several files can share the same day.
# MATERIAL_FILES lists the 1-based file numbers for each material.
MATERIAL = "soil"

MATERIAL_FILES = {
    "soil":               list(range(1, 13)),   # files 1-12  (Oct 21-23 2025, Feb 02-03 2026)
    "decomposed_granite": list(range(13, 18)),  # files 13-17 (Feb 04 + Feb 11 2026, Site 1)
    "sand":               list(range(18, 24)),  # files 18-23 (Feb 11-13 2026, Site 2)
}
MATERIAL_FILES["all"] = list(range(1, 24))       # every file

if MATERIAL not in MATERIAL_FILES:
    raise ValueError(f"MATERIAL must be one of {sorted(MATERIAL_FILES)}, got {MATERIAL!r}")

"""""""""""""""""""""""""""""""""""""""October 2025: Site 1: Soil"""""""""""""""""""""""""""""""""

################## October 21 2025: Site 1: Soil

Data_tasks_1 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_21_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

################## October 22 2025: Site 1: Soil

Data_tasks_2 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_22_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)
Data_tasks_3 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_22_Tasks_2.xlsx'),
    sheet_name="Sheet1"
)
Data_tasks_4 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_22_Tasks_3.xlsx'),
    sheet_name="Sheet1"
)
Data_tasks_5 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_22_Tasks_4.xlsx'),
    sheet_name="Sheet1"
)
Data_tasks_6 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_22_Tasks_5.xlsx'),
    sheet_name="Sheet1"
)

################## October 23 2025: Site 1: Soil

Data_tasks_7 = pd.read_excel(
    os.path.join(DATA_DIR, 'Oct_23_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)


"""""""""""""""""""""""""""""""""""""""February 2026: Site 1: Soil"""""""""""""""""""""""""""""""""

################## February 02 2026: Site 1: Soil

Data_tasks_8 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_02_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_9 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_02_Tasks_2.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_10 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_02_Tasks_3.xlsx'),
    sheet_name="Sheet1"
)



################## February 03 2026: Site 1: Soil


Data_tasks_11 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_03_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_12 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_03_Tasks_2.xlsx'),
    sheet_name="Sheet1"
)


################## February 04 2026: Site 1: Decomposed Granite


Data_tasks_13 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_04_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_14 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_04_Tasks_2.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_15 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_04_Tasks_3.xlsx'),
    sheet_name="Sheet1"
)

################## February 11 2026: Site 1: Decomposed Granite

Data_tasks_16 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_11_Tasks_2.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_17 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_11_Tasks_3.xlsx'),
    sheet_name="Sheet1"
)

################## February 11 2026: Site 2: Sand


Data_tasks_18 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_11_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

################## February 12 2026: Site 2: Sand


Data_tasks_19 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_12_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_20 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_12_Tasks_2.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_21 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_12_Tasks_3.xlsx'),
    sheet_name="Sheet1"
)

Data_tasks_22 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_12_Tasks_4.xlsx'),
    sheet_name="Sheet1"
)

################## February 13 2026: Site 2: Sand

Data_tasks_23 = pd.read_excel(
    os.path.join(DATA_DIR, 'Feb_13_Tasks_1.xlsx'),
    sheet_name="Sheet1"
)

Battery_cap = 14.8

"""""""""""""""""""""""""""""""""""""""HELPER FUNCTIONS"""""""""""""""""""""""""""""""""

def prepare_task_data(df):
    """
    Keep needed columns, parse timestamps, compute duration in seconds,
    and remove negative durations.
    """
    df_clean = df[['Start time (actual)', 'End time (actual)', 'Activity', 'SoC']].copy()

    df_clean["Start time (actual)"] = pd.to_datetime(df_clean["Start time (actual)"], errors="coerce")
    df_clean["End time (actual)"]   = pd.to_datetime(df_clean["End time (actual)"], errors="coerce")

    df_clean["duration_s"] = (
        df_clean["End time (actual)"] - df_clean["Start time (actual)"]
    ).dt.total_seconds()

    df_clean.loc[df_clean["duration_s"] < 0, "duration_s"] = np.nan

    return df_clean


def build_equations_from_tasks(df_clean, battery_cap):
    """
    Walk rows accumulating activity time into a bucket. Emit one equation when
    cumulative |ΔSoC| from the bucket's anchor row reaches MIN_DELTA_SOC, then
    start a fresh bucket anchored at the current row.

    Cumulative-bucket attribution preserves total energy when MIN_DELTA_SOC > 1:
    1% drops that don't individually clear the threshold are merged into the
    next equation along with their activity time, so no SoC drop is discarded.

    Setting MIN_DELTA_SOC = 1 reproduces one-equation-per-step.
    """
    A_rows_all = []
    b_rows_all = []

    if len(df_clean) == 0:
        return A_rows_all, b_rows_all

    # Anchor the first bucket at the first valid SoC reading.
    bucket_start = 0
    while bucket_start < len(df_clean) and pd.isna(df_clean.iloc[bucket_start]["SoC"]):
        bucket_start += 1
    if bucket_start >= len(df_clean):
        return A_rows_all, b_rows_all
    bucket_anchor_soc = df_clean.iloc[bucket_start]["SoC"]

    for j in range(bucket_start + 1, len(df_clean)):
        soc_now = df_clean.iloc[j]["SoC"]
        if pd.isna(soc_now):
            continue

        cumulative_delta = soc_now - bucket_anchor_soc
        if abs(cumulative_delta) < MIN_DELTA_SOC:
            continue

        df_slice = df_clean.iloc[bucket_start:j + 1]

        total_s_Digging    = df_slice.loc[df_slice["Activity"] == "Digging",    "duration_s"].sum()
        total_s_Loading    = df_slice.loc[df_slice["Activity"] == "Loading",    "duration_s"].sum()
        total_s_Swinging   = df_slice.loc[df_slice["Activity"] == "Swinging",   "duration_s"].sum()
        total_s_Traveling  = df_slice.loc[df_slice["Activity"] == "Traveling",  "duration_s"].sum()
        total_s_Idling     = df_slice.loc[df_slice["Activity"] == "Idling",     "duration_s"].sum()
        total_s_Mixing     = df_slice.loc[df_slice["Activity"] == "Mixing",     "duration_s"].sum()

        total_energy = -cumulative_delta * battery_cap / 100

        # Loading and Swinging are kept as separate columns here; they are
        # clubbed into a single regressor after the collinearity check below.
        A_row = [
            total_s_Digging/3600,
            total_s_Loading/3600,
            total_s_Swinging/3600,
            total_s_Traveling/3600,
            total_s_Idling/3600,
            total_s_Mixing/3600,
        ]

        A_rows_all.append(A_row)
        b_rows_all.append([total_energy])

        bucket_start = j + 1
        bucket_anchor_soc = soc_now

    return A_rows_all, b_rows_all

# Use MOSEK when a licensed install is available; otherwise fall back to
# Clarabel, the open-source QP solver bundled with cvxpy, so the script runs
# out of the box from a fresh clone.
SOLVER = "MOSEK" if "MOSEK" in cp.installed_solvers() else "CLARABEL"

def solve_activity_power(A, b, n, W, reg_param):

    z = cp.Variable(n, nonneg=True)
    constraints = [z[3] == 0]   # Idling power fixed to 0

    objective = cp.Minimize(cp.sum_squares(cp.sqrt(W) @ (A @ z - b)) + reg_param*cp.sum_squares(z))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=SOLVER)

    return z.value, objective.value


def compute_weights(b_arr, scheme):
    """
    Build per-equation weights from |b| (proportional to |ΔSoC|). The returned
    vector is normalized to mean 1 so the residual term and the L2 regularization
    term stay on the same effective scale across schemes — no need to retune
    reg_param when switching schemes.

    Schemes
    -------
    "uniform"         : w_i = 1                              (no weighting)
    "linear"          : w_i = |b_i|                          (sharp; ~6x for 6% vs 1% ΔSoC)
    "bounded_median"  : w_i = min(|b_i| / median(|b|), 1)    (capped above the median; only
                                                              sub-median equations are penalized)
    "quadratic"       : w_i = b_i^2                          (very sharp; ~36x for 6% vs 1%)
    """
    b_abs = np.abs(b_arr)
    if scheme == "uniform":
        w = np.ones_like(b_abs)
    elif scheme == "linear":
        w = b_abs.copy()
    elif scheme == "bounded_median":
        ref = np.median(b_abs)
        w = np.minimum(b_abs / ref, 1.0)
    elif scheme == "quadratic":
        w = b_abs ** 2
    else:
        raise ValueError(f"Unknown WEIGHT_SCHEME: {scheme!r}")
    return w / w.mean()


"""""""""""""""""""""""""""""""""""""""PREPARE DATA"""""""""""""""""""""""""""""""""

Data_tasks_clean_1 = prepare_task_data(Data_tasks_1)        # Oct 21, 2025: Soil

Data_tasks_clean_2 = prepare_task_data(Data_tasks_2)
Data_tasks_clean_3 = prepare_task_data(Data_tasks_3)
Data_tasks_clean_4 = prepare_task_data(Data_tasks_4)
Data_tasks_clean_5 = prepare_task_data(Data_tasks_5)   
Data_tasks_clean_6 = prepare_task_data(Data_tasks_6)   
Data_tasks_clean_7 = prepare_task_data(Data_tasks_7)        # Oct 22-23, 2025: Soil


Data_tasks_clean_8 = prepare_task_data(Data_tasks_8)   
Data_tasks_clean_9 = prepare_task_data(Data_tasks_9)   
Data_tasks_clean_10 = prepare_task_data(Data_tasks_10)      # Feb 02, 2026: Soil

Data_tasks_clean_11 = prepare_task_data(Data_tasks_11)      
Data_tasks_clean_12 = prepare_task_data(Data_tasks_12)      # Feb 03, 2026: Soil

Data_tasks_clean_13 = prepare_task_data(Data_tasks_13)
Data_tasks_clean_14 = prepare_task_data(Data_tasks_14)
Data_tasks_clean_15 = prepare_task_data(Data_tasks_15)      # Feb 04, 2026: Decomposed Granite

Data_tasks_clean_16 = prepare_task_data(Data_tasks_16)
Data_tasks_clean_17 = prepare_task_data(Data_tasks_17)      # Feb 11, 2026: Decomposed Granite

Data_tasks_clean_18 = prepare_task_data(Data_tasks_18)      # Feb 11, 2026: Sand

Data_tasks_clean_19 = prepare_task_data(Data_tasks_19)
Data_tasks_clean_20 = prepare_task_data(Data_tasks_20)   
Data_tasks_clean_21 = prepare_task_data(Data_tasks_21)   
Data_tasks_clean_22 = prepare_task_data(Data_tasks_22)      # Feb 12, 2026: Sand

Data_tasks_clean_23 = prepare_task_data(Data_tasks_23)      # Feb 13, 2026: Sand





Data_tasks_clean = [Data_tasks_clean_1,Data_tasks_clean_2, Data_tasks_clean_3, Data_tasks_clean_4, Data_tasks_clean_5, Data_tasks_clean_6, 
                    Data_tasks_clean_7,Data_tasks_clean_8, Data_tasks_clean_9, Data_tasks_clean_10, Data_tasks_clean_11, Data_tasks_clean_12, 
                    Data_tasks_clean_13,Data_tasks_clean_14, Data_tasks_clean_15, Data_tasks_clean_16, Data_tasks_clean_17, Data_tasks_clean_18, 
                    Data_tasks_clean_19,Data_tasks_clean_20, Data_tasks_clean_21, Data_tasks_clean_22, Data_tasks_clean_23 
                    ]



"""""""""""""""""""""""""""""""""""""""FORM EQUATIONS FROM SELECTED TASK FILES"""""""""""""""""""""""""""""""""

# Automatically pick the task files for the selected MATERIAL. Data_tasks_clean
# holds all 23 task files in order, so file f (1-based) is Data_tasks_clean[f - 1].
selected_files = MATERIAL_FILES[MATERIAL]
all_task_dfs = [Data_tasks_clean[f - 1] for f in selected_files]
print(f"MATERIAL = {MATERIAL!r}  ->  using task files {selected_files} ({len(all_task_dfs)} files)")



A = []
b = []

for df_clean in all_task_dfs:
    A_part, b_part = build_equations_from_tasks(df_clean, Battery_cap)
    A.extend(A_part)
    b.extend(b_part)
    
df = pd.concat(all_task_dfs)
unique_tasks = df["Activity"].unique()
print(f"\n\n************OUTPUT************\n\n")

print(f"The number of unique tasks are in the dataset are: {unique_tasks}")
print(f"Equations built: {len(A)}  (MIN_DELTA_SOC = {MIN_DELTA_SOC} %)")

"""""""""""""""""""""""""""""""""""""""DEFINING MATRIX AND VECTOR"""""""""""""""""""""""""""""""""

# Unclubbed design matrix: Digging, Loading, Swinging, Traveling, Idling, Mixing.
A_raw = np.asarray(A)
b = np.asarray(b).reshape(-1)

# Pearson correlation among the unclubbed regressors (Table I in the paper).
# Loading and Swinging are strongly correlated, motivating the clubbing below.
df_raw = pd.DataFrame(A_raw)
df_raw.columns = ['Digging', 'Loading', 'Swinging', 'Traveling', 'Idling', 'Mixing']
corr_raw = df_raw.corr()
print("\n---- Correlation matrix BEFORE clubbing Loading & Swinging ----")
print(corr_raw.to_string(float_format=lambda x: f"{x:6.2f}"))
plt.figure()
sns.heatmap(corr_raw, annot=True, cmap='coolwarm')
plt.xticks(rotation=45)
plt.title("Regressor correlation (before clubbing)")
plt.show()

# Club Loading and Swinging into a single regressor.
A = np.column_stack([
    A_raw[:, 0],                 # Digging
    A_raw[:, 1] + A_raw[:, 2],   # Loading + Swinging
    A_raw[:, 3],                 # Traveling
    A_raw[:, 4],                 # Idling
    A_raw[:, 5],                 # Mixing
])

m, n = A.shape

# Which activities actually occur in the dataset. A column that is all-zero
# (e.g. Mixing on soil-only days) carries no data, so we suppress those
# activities from the printed results. Index-aligned with `labels` below.
col_present = A.sum(axis=0) > 1e-12

Time_Digging = np.sum(A[:, [0]], axis = 0); #axis = 0 means across rows
Time_Loading_Swinging = np.sum(A[:, [1]], axis = 0);
Time_Traveling = np.sum(A[:, [2]], axis = 0);
Time_Mixing = np.sum(A[:, [4]], axis = 0);
Time_Idling = np.sum(A[:, [3]], axis = 0);

Time_all = Time_Digging + Time_Loading_Swinging + Time_Traveling + Time_Mixing + Time_Idling;

df = pd.DataFrame(A)
df.columns = ['Digging', 'Loading+Swinging', 'Traveling', 'Idling', 'Mixing']
corr = df.corr()
print("\n---- Correlation matrix AFTER clubbing Loading & Swinging ----")
print(corr.to_string(float_format=lambda x: f"{x:6.3f}"))
plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.xticks(rotation=45)
plt.title("Regressor correlation (after clubbing)")
plt.show()




"""""""""""""""""""""""""""""""""""""""REPEATED K-FOLD CV"""""""""""""""""""""""""""""""""

reg_param = 0e-3

labels = ['Digging', 'Loading+Swinging',
          'Traveling', 'Idling', 'Mixing']

metric_keys = ["mae", "rmse", "mape", "nmae"]

# K=5 folds × n_repeats=200 = 1000 fits. Each equation lands in the test fold
# exactly n_repeats=200 times — uniform coverage.
KF_K = 5
KF_REPEATS = 200

all_indices = np.arange(m)
rkf = RepeatedKFold(n_splits=KF_K, n_repeats=KF_REPEATS, random_state=0)

metrics_kf = {k: [] for k in metric_keys}
z_samples_kf = []
# Track per-equation predictions across the folds where each equation was in
# the test fold, so we can plot observed vs median-predicted with empirical bands.
test_preds_per_eq_kf = [[] for _ in range(m)]

for idx_tr, idx_te in rkf.split(all_indices):
    A_tr, A_te = A[idx_tr], A[idx_te]
    b_tr, b_te = b[idx_tr], b[idx_te]

    # R4: per-equation weights drive ΔSoC-based confidence. Scheme is set
    # via WEIGHT_SCHEME at the top of this file (see compute_weights).
    w_tr = compute_weights(b_tr, WEIGHT_SCHEME)
    W_tr = np.diag(w_tr)
    z_s, _ = solve_activity_power(A_tr, b_tr, n, W_tr, reg_param)
    b_pred = A_te @ z_s

    mae  = mean_absolute_error(b_te, b_pred)
    rmse = np.sqrt(mean_squared_error(b_te, b_pred))
    mape = np.mean(np.abs((b_te - b_pred) / b_te)) * 100
    nmae = mae / np.mean(np.abs(b_te)) * 100

    metrics_kf["mae"].append(mae)
    metrics_kf["rmse"].append(rmse)
    metrics_kf["mape"].append(mape)
    metrics_kf["nmae"].append(nmae)
    z_samples_kf.append(z_s)

    for local_i, eq_i in enumerate(idx_te):
        test_preds_per_eq_kf[eq_i].append(b_pred[local_i])

z_samples_kf = np.array(z_samples_kf)   # shape (KF_K * KF_REPEATS, n)


"""""""""""""""""""""""""""""""""""""""PRINTING"""""""""""""""""""""""""""""""""

print(f"\n---- Repeated {KF_K}-fold CV (n_repeats = {KF_REPEATS}, total fits = {KF_K * KF_REPEATS}, weight scheme = {WEIGHT_SCHEME!r}) ----")
print("Test-set metrics across folds (mean ± SD):")
for k in metric_keys:
    vals = np.array(metrics_kf[k])
    units = "kWh" if k in ("mae", "rmse") else "%"
    print(f"  {k.upper():5s} = {vals.mean():7.3f} ± {vals.std():.3f}  {units}")

print("\nCoefficients across folds (mean ± SD, 95% empirical interval):")
for i, lab in enumerate(labels):
    if not col_present[i]:
        continue
    col = z_samples_kf[:, i]
    lo, hi = np.percentile(col, [2.5, 97.5])
    print(f"  {lab:30s}  {col.mean():6.2f} ± {col.std():5.2f} kW    [{lo:5.2f}, {hi:5.2f}]")

print(f"\nActivity time share (full dataset):")
if col_present[0]: print(f"  Digging:           {Time_Digging[0]*100/Time_all[0]:.0f}%")
if col_present[1]: print(f"  Loading+Swinging:  {Time_Loading_Swinging[0]*100/Time_all[0]:.0f}%")
if col_present[2]: print(f"  Traveling:         {Time_Traveling[0]*100/Time_all[0]:.0f}%")
if col_present[4]: print(f"  Mixing:            {Time_Mixing[0]*100/Time_all[0]:.0f}%")
if col_present[3]: print(f"  Idling:            {Time_Idling[0]*100/Time_all[0]:.0f}%")



"""""""""""""""""""""""""""""""""""""""PLOTS"""""""""""""""""""""""""""""""""

# Observed vs median-predicted across the KF_REPEATS folds in which each
# equation was held out. Vertical bars show the 2.5–97.5 percentile prediction band.
median_preds, lower_preds, upper_preds, included_idx = [], [], [], []
for i in range(m):
    preds = test_preds_per_eq_kf[i]
    if len(preds) > 0:
        median_preds.append(np.median(preds))
        lower_preds.append(np.percentile(preds, 2.5))
        upper_preds.append(np.percentile(preds, 97.5))
        included_idx.append(i)
included_idx = np.array(included_idx)
median_preds = np.array(median_preds)
lower_preds  = np.array(lower_preds)
upper_preds  = np.array(upper_preds)
obs = b[included_idx]
yerr = np.vstack([median_preds - lower_preds, upper_preds - median_preds])

plt.figure(figsize=(6, 6))
plt.errorbar(obs, median_preds, yerr=yerr, fmt='o', ecolor='gray',
             capsize=2, label='Median ± 95% band')
lo_lim = min(obs.min(), lower_preds.min())
hi_lim = max(obs.max(), upper_preds.max())
plt.plot([lo_lim, hi_lim], [lo_lim, hi_lim], '--', color='black', label='y = x')
plt.xlabel("Observed energy (kWh)")
plt.ylabel("Predicted energy (kWh, median across folds)")
plt.title(f"Observed vs Predicted (repeated {KF_K}-fold CV, {KF_REPEATS} repeats)")
plt.legend()
plt.tight_layout()
plt.show()



if sys.flags.interactive == 0 and plt.get_fignums():
    plt.ioff()
    plt.show()