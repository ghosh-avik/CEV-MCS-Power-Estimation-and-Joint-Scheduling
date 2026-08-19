# Power Estimation and Optimal Work–Charging Scheduling of Construction Electric Vehicles via Mobile Charging Stations

Code and data accompanying the paper:

> A. Ghosh\*, A. Taşçıkaraoğlu\*, D. Rojas, M. A. Beyazıt, M. R. Salehizadeh, K. Chia, S. Doppelt, M. Ferry, J. Kleissl, S. Dey, and Y. Shi,
> **"Power Estimation and Optimal Work–Charging Scheduling of Construction Electric Vehicles via Mobile Charging Stations."**
> (\* equal contribution) — arXiv link coming soon. The paper has been submitted for publication in IEEE Transactions on Smart Grid.
>
> This repository (`github.com/ghosh-avik/CEV-MCS-Power-Estimation-and-Joint-Scheduling`) is the "dataset and scripts" repository linked in the paper's abstract and cited as its reference [27].

Construction electric vehicles (CEVs) are a clean alternative to diesel construction equipment, but their adoption is held back by sparse onsite charging infrastructure, limited CEV mobility, and poor understanding of their power consumption. This repository addresses both gaps with two coupled components:

1. **CEV Subactivity Power Estimation** — using field data from a real construction demonstration at UC San Diego (a JCB 19C-1E compact electric excavator), we estimate the average power drawn by each work *subactivity* (digging, loading, swinging, traveling, idling, mixing) from manually labeled video synchronized with coarse battery state-of-charge (SOC) telematics, via constrained non-negative least squares. **The raw dataset is released here** — to our knowledge the first public per-subactivity power dataset for CEVs.
2. **Joint Scheduling** — using those subactivity power estimates, a mixed-integer program (MIP) jointly optimizes *when and which subactivities each CEV performs* together with *where, when, and how much its mobile charging station (MCS) charges from the grid and discharges to the CEVs*, under energy and demand charges, carbon emissions, missed-work penalties, MCS travel costs, and the physical/operational constraints of both vehicle types.

## Repository structure

```
├── CEV Subactivity Power Estimation/
│   ├── Raw Data/                  23 task-recording Excel files (the released dataset)
│   ├── Power Estimation.py        Full estimation + validation pipeline
│   └── requirements.txt           Python dependencies for this folder
│
└── Joint Scheduling/
    ├── Scenario 1/ … Scenario 4/  One folder per case-study scenario (paper Sec. IV)
    │   ├── mcs_optimization_main_v4_real.jl     Driver script (identical in all scenarios)
    │   ├── helper functions/
    │   │   ├── DataLoader_v4_real.jl            Reads the scenario's csv_files/
    │   │   ├── MCS_OPTIMAL_v4_real.jl           Proposed joint optimization
    │   │   ├── MCS_Reference_v4_real.jl         Rule-based reference CEV scheduler
    │   │   └── MCS_B1_B2_v4_real.jl             Paper baselines B1 and B2
    │   └── simple_dataset/
    │       ├── csv_files/          Scenario inputs (topology, EV/MCS specs, prices, …)
    │       └── results/            Solved runs used in the paper (included)
    ├── scenario_1_comparison.py                 Reproduces the Scenario 1 operation figure (Fig. 4)
    ├── scenario_2_comparison.py                 Reproduces the Scenario 2 operation figure (Fig. 8)
    ├── scenario_1_mip_convergence_analysis.py   Reproduces the solver-convergence figure (Fig. 7, Appendix F)
    └── requirements.txt                         Python dependencies for the figure scripts
```

All code files are **identical across the four scenario folders**; scenarios differ only in their `simple_dataset/csv_files/` inputs (network topology, number of CEVs/MCSs, travel times, work requirements).

---

## Part 1 — CEV Subactivity Power Estimation

### The problem

Battery SOC telematics are coarse (integer percent, sparse in time), while a CEV switches subactivities every few seconds. Over each *observation window* between SOC readings, the measured energy drop must equal the sum of (unknown per-subactivity power) × (observed per-subactivity duration). Stacking windows gives an over-determined linear system `D p = b + ε`, solved for the power vector `p` by constrained non-negative least squares with idling power pinned to zero (paper Sec. II). Windows are built *quantization-aware* (a new window only after the cumulative SOC drop reaches τ = 3 %) so that each equation's signal dominates the ±0.5 % SOC rounding error.

### The data

`Raw Data/` contains 23 task-recording Excel files from two field campaigns on the UC San Diego campus (October 2025 and February 2026), spanning ≈20 hours of operation across three working materials:

| Material | Files | Days |
|---|---|---|
| Soil | 12 | Oct 21–23, 2025 + Feb 02–03, 2026 (Site 1) |
| Decomposed granite | 5 | Feb 04 + Feb 11, 2026 (Site 1) |
| Sand | 6 | Feb 11–13, 2026 (Site 2) |

Each row is one manually labeled subactivity segment: `Activity` (one of the six Table V categories: Digging, Loading, Swinging, Traveling, Idling, Mixing — the last occurring only in sand work), actual start/end timestamps at one-second resolution, and the concurrent battery `SoC` [%].

### Running it

```bash
cd "CEV Subactivity Power Estimation"
pip install -r requirements.txt
python "Power Estimation.py"
```

Select the material at the top of the script (`MATERIAL = "soil"`, `"decomposed_granite"`, `"sand"`, or `"all"`). The script prints the regressor correlation matrices before and after clubbing Loading+Swinging (Table I), the repeated 5-fold cross-validated subactivity powers and held-out error metrics, and the activity time shares — reproducing **Tables I–III** for soil and **Tables VI–VIII** (Appendix D) for decomposed granite and sand.

The quadratic program is solved with MOSEK when a licensed install is detected, and otherwise falls back automatically to **Clarabel** (open-source, bundled with cvxpy) — both produce identical results, so no commercial license is needed.

---

## Part 2 — Joint MCS–CEV Scheduling

### The problem

An MCS (CleanGen J250 — a towable 250 kWh battery) shuttles between grid-connection nodes, where it charges, and construction nodes, where it discharges into working CEVs. Because subactivity power demands differ substantially (≈4.8 kW digging vs ≈3.2 kW loading+swinging on soil), *reordering the work itself* reshapes the CEV demand profile and hence when and how much the MCS must draw from the grid. The MIP (paper Sec. III) co-optimizes CEV work schedules, CEV charging intervals, and MCS routing/charging/discharging over a 24 h horizon in 15-min steps, minimizing energy costs, non-coincident and on-peak demand charges, carbon costs, missed-work penalties, and MCS travel labor.

### Strategies

Each scenario driver can run four strategies, selected via the `MCS_OPTIMIZER_CHOICE` environment variable (or the manual toggle near the top of the driver):

| Choice | Paper name | What it does |
|---|---|---|
| `Reference` | — | Rule-based "work-to-minimum / charge-to-full" reference scheduler. Exports the frozen CEV schedule (`results/ref_schedule_Reference.csv`) consumed by B1/B2. Not compared directly in Table IV. |
| `B1` | Baseline 1 | CEV work **and** charging intervals pinned to the reference; only MCS decisions are optimized (conventional MCS scheduling with a given demand profile). |
| `B2` | Baseline 2 | Only the CEV work schedule is pinned; CEV charging and all MCS decisions are free. |
| `OPTIMAL` | Proposed | Full joint decision space (default). |

### Running it

Requires [Julia](https://julialang.org/) (tested with ≥ 1.10) with `JuMP`, `HiGHS`, `Plots`, `DataFrames`, `CSV`:

```julia
using Pkg; Pkg.add(["JuMP", "HiGHS", "MathOptInterface", "Plots", "DataFrames", "CSV"])
```

Then, from a scenario folder, run the ladder in order (Reference must come first on a fresh clone only if you want to regenerate the frozen schedule — the solved `ref_schedule_Reference.csv` files are already included):

```bash
cd "Joint Scheduling/Scenario 1"
MCS_OPTIMIZER_CHOICE=Reference julia mcs_optimization_main_v4_real.jl
MCS_OPTIMIZER_CHOICE=B1        julia mcs_optimization_main_v4_real.jl
MCS_OPTIMIZER_CHOICE=B2        julia mcs_optimization_main_v4_real.jl
MCS_OPTIMIZER_CHOICE=OPTIMAL   julia mcs_optimization_main_v4_real.jl
```

Each run writes a timestamped folder `<CHOICE>_Site_<n>_MCS_<m>_CEV_<e>_<timestamp>/` under `simple_dataset/results/` containing the cost/KPI breakdown, power and SOE profiles, the solved CEV schedule, the MIP convergence trace, and summary plots. The solver time limit defaults to 1 hour (as in Table IV) and is overridable, e.g. `MCS_TIME_LIMIT_SEC=18000` for the 5-hour extended Scenario 4 re-solve of Appendix F.

> **Caution:** every `Reference` run *overwrites* `results/ref_schedule_Reference.csv`, the frozen CEV schedule that B1 and B2 pin to. If you run `Reference` with a reduced time limit (e.g. a quick smoke test via `MCS_TIME_LIMIT_SEC`), the degraded schedule will shift subsequent B1/B2 results away from Table IV. With the included schedule, B1/B2 reproduces its Table IV cost exactly.

The runs used in the paper are included in each scenario's `results/` folder, so all figures and Table IV values can be verified without re-solving. Solving uses HiGHS (open source); every instance except the proposed strategy in Scenario 4 proves optimality within the 1-hour limit (see Appendix F for why Scenario 4's residual gap does not affect the reported cost).

### Figure scripts

```bash
cd "Joint Scheduling"
pip install -r requirements.txt
python scenario_1_comparison.py                 # Scenario 1 operation panel (paper Fig. 4)
python scenario_2_comparison.py                 # Scenario 2 operation panel (paper Fig. 8)
python scenario_1_mip_convergence_analysis.py   # Scenario 4 MIP convergence (paper Fig. 7, Appendix F)
```

These read only the included `results/` CSVs, write their outputs into `Joint Scheduling/` (`scenario1_operation_zoom.png`, `scenario2_operation_zoom.png`, `scenario_1_mip_convergence_normalized.png` plus a summary CSV), and print the per-strategy demand peaks (which reproduce the NCDC values in Table IV, e.g. Scenario 1: 1.95/2.44/1.82 kW × $20.12/kW ≈ $39/$49/$37).

---

## Reproducibility summary

| Paper result | How to reproduce |
|---|---|
| Tables I–III (soil power estimation) | `Power Estimation.py` with `MATERIAL = "soil"` |
| Tables VI–VIII (decomposed granite, sand — Appendix D) | same script, `MATERIAL = "decomposed_granite"` / `"sand"` |
| Table IV (cost breakdown, 4 scenarios) | run the strategy ladder per scenario, or inspect the included `results/` runs |
| Figs. 4 and 8 (scenario operation panels) | `scenario_1_comparison.py`, `scenario_2_comparison.py` |
| Fig. 7 (solver convergence, Appendix F) | `scenario_1_mip_convergence_analysis.py`; extended run via `MCS_TIME_LIMIT_SEC=18000` |

## Citation

If you use this code or dataset, please cite the paper:

```bibtex
@misc{ghosh2026cev,
  title         = {Power Estimation and Optimal Work--Charging Scheduling of
                   Construction Electric Vehicles via Mobile Charging Stations},
  author        = {Ghosh, Avik and Ta{\c{s}}{\c{c}}{\i}karao{\u{g}}lu, Ak{\i}n and
                   Rojas, Daniela and Beyaz{\i}t, Muhammed A. and
                   Salehizadeh, Mohammad Reza and Chia, Keaton and Doppelt, Sasha and
                   Ferry, Michael and Kleissl, Jan and Dey, Sujit and Shi, Yuanyuan},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/XXXX.XXXXX},
  note          = {Submitted for publication in IEEE Transactions on Smart Grid.
                   arXiv identifier to be updated.}
}
```

## License

Code and data are released under the [MIT License](LICENSE).

## Acknowledgment

This work was supported by the California Energy Commission under the award number EPC-24-031. A. Taşçıkaraoğlu was also supported by the TÜBİTAK 2219 International Postdoctoral Research Fellowship Program and the TÜBA Distinguished Young Scientist Award. M. A. Beyazıt was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) Directorate of Science Fellowships and Grant Programmes (BİDEB) 2211 National PhD Scholarship Program. M. R. Salehizadeh was supported through the QRRRF-funded Driving Resilience project (CQU.0001.2324D.RFI).

The authors gratefully acknowledge the help of Shubhan Mital in manually labeling the dataset for the excavator subactivity power consumption analysis. During the preparation of this work the authors used Claude to help with the visualization of the results figures. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the article.
