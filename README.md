# RAAH-ZT: reproduction code (first revision)

Code, fixed protocols and result files for the first revision of
*RAAH-ZT: Risk-Aware Adaptive Hysteresis for Stable Zero-Trust Access Control in IoT Networks*
(IEEE Internet of Things Journal, manuscript IoT-76550-2026).

All rates in the CSV/JSON files are fractions; the paper and the supplement show them in percent.

## Layout

| Folder | Content |
|---|---|
| `data/` | Validation and test timelines (`event_id`, `label`, `group`, `timestamp`, `trust`, `base_action`) derived from UNSW-NB15. Size and SHA-256 are checked on every load. |
| `frozen/` | The fixed protocol (`protocol.lock.json`), the validation selection (`validation_selection.lock.json`) and the unchanged runner of the submitted version (`run_experiment.py`). |
| `frozen/original/` | Results of the submitted version, used to check that the new code reproduces them exactly. |
| `r1/` | Scripts for all analyses added in the revision, and `r1_lib.py` with the shared delayed-feedback engine. |
| `r1/results/` | All result files of the revision (CSV/JSON/NPZ). |
| `figures_new/` | Figures of the revision (vector PDF). |
| `supplement/` | LaTeX sources of the supplement; `gen_s7.py` builds Section S7 from `r1/results/`. |

## Setup

Python 3.13 on Windows or Linux:

```bash
pip install -r requirements.txt
```

## Quick check (about 2 minutes)

```bash
cd r1
python -c "import r1_lib as L; v,t=L.load(); a,s=L.actions_for(t); g=L.R.global_order(t,a,s); print(L.check_engine(t,g))"
```

The printed value is the largest difference between the engine and the submitted feedback rows; it must be `0.0`.

## Reproducing the results

Run the scripts from inside `r1/`. Each one writes to `r1/results/`.

| Script | Results | Paper |
|---|---|---|
| `exp_static.py` | A1 tie permutations, A2 inactivity timeouts, A7 candidate frontier, A8 Theorem 4 | Sec. V-C, V-D, Table IV |
| `exp_dynamic.py` | A3 fallback rules, A4 label-noise curve, A5 switch decomposition, A6 query rules | Sec. V-E, Table V, Fig. 8 |
| `exp_robust.py b1` / `exp_robust.py b2` | B1 bootstrap frontier, B2 replicates | Sec. V-C, Fig. 6; Sec. V-E |
| `exp_rerun75.py` | C1 all 75 feedback rows with the revised fallback | Sec. V-E, Fig. 7 |
| `exp_score_shift_R1.py` | C2 score-calibration shift with the revised fallback | Sec. V-E1, Fig. 9 |
| `exp_misspec_R1.py` | C3 misspecified flip rate | Sec. V-E |
| `exp_timing_local.py` | A9 timing per layer, A11 local MFA windows | Sec. V-H, Sec. V-E2 |
| `exp_causal_features.py` | A10 evaluator with start-time features | Sec. V-D |
| `make_figures.py` | Figs. 2, 3, 6--10 | |
| `fig1_final.py` | Fig. 1 | |

`supplement/gen_s7.py` rebuilds the tables of Supplement S7 from these results.

### A10 (start-time features)

This analysis refits the HGB evaluator and therefore needs the raw UNSW-NB15 CSV files
(available from the dataset authors) and the fitting code of the earlier reconstruction.
Set `RAAH_UNSW_WORKDIR` to that working directory before running `exp_causal_features.py`.
Its per-flow prediction files (about 135 MB) are not stored here; all other A10 results are in `r1/results/A10_start_features/`.

## Notes

- The study is a retrospective replay of completed flows with recommendation-level proxy metrics; see the paper for its scope.
- The submitted results are kept unchanged in `frozen/original/` so that each new run can be compared with them.
- In `frozen/protocol.lock.json` only the two input paths were made relative; `frozen/PROVENANCE.md` lists the hashes of the original and the public file.
- License: MIT (see `LICENSE`).
