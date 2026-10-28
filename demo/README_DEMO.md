# Exploratory demo: accept or abstain for one partial 12-lead ECG

> **Public-release note (28 Oct 2026).** This file is the project's internal demo record, kept for provenance.
> * **Not in this repository:** `demo/examples/`, `demo/example_outputs/` and `demo/verification/`. They are derived from
>   PTB-XL record 17 or from development-role data, and no patient recording is published here.
> * **Instead:** use the synthetic inputs in `examples/synthetic/`, for example
>   `demo/infer_partial_ecg.py --csv examples/synthetic/synthetic_cand01.csv OUT_DIR`.
> * **The `--wfdb` mode** needs your own PTB-XL copy and the frozen split manifest, which is not distributed.
>
> See `SUPERVISOR_DEMO_GUIDE.md`.

> **Status: EXPLORATORY DEMO, NOT VALIDATED.** This is a research prototype. It has no external or clinical validation and
> is not a medical device. The gate was evaluated once, on PTB-XL fold 10, with synthetic whole-lead missingness (final
> report, Sections 8–10). An ACCEPT is **not** evidence that the reconstruction is diagnostically faithful. Gate B withholds
> most bundle-branch-block ECGs, especially CLBBB.

Created 2026-10-24 (stage 07 of M4b-1). Nothing in `src/`, `configs/`, `tests/`, the models, calibrators, locks or
thresholds was changed. The code-tree hash `7cb6c7f9…` is unchanged, because `demo/` is outside the code-tree patterns.

## 1. Did an inference path exist before? (Task 4 assessment)

**Not end to end.** Every existing command-line entry point works on PTB-XL roles taken from the frozen split manifest;
none accepts a user-supplied ECG file. The fold-10 pathway (`src/eval/fold10_pathway.py`) also needs the complete ECG,
because it computes D1. All the pieces existed as tested library functions:

| Step | Existing function | Frozen asset |
| --- | --- | --- |
| Lead mask, 258 lead sets | `src/data/missingness.py` (`lead_mask`, `l2g_lead_combinations`) | — |
| Preprocessing (high-pass 0.5 Hz, notch 50 Hz, 500 → 250 Hz, 2,048-sample window, per-lead min–max) | `src/data/preprocess.py` (`filter_resample`, `crop_normalize`) | `configs/preprocessing.yaml` |
| L2G reconstruction with exact copy-through | `src/eval/generate_d1.py` (`reconstruct`) | `artifacts/runs/l2g_v0_1_run02/best.pt` (`d515875a…`) |
| G1/G2 gate features | `src/gate/features.py` (`row_features`) | `configs/gate_m4b_v0_2.yaml` (`80fdd7a4…`) |
| Gate B score, Platt calibration | `src/gate/m4b1.py` (`load_models`, `load_calibrators`, `method_matrix`, `score`, `calibrated`) | models and calibrators checked against their stage records |
| Locked threshold | `src/eval/primary_decision.py` (`threshold_from_record`) | `artifacts/locks/threshold_lock_v0_2.json` (`bc69c301…`) |

`demo/infer_partial_ecg.py` only calls these functions, in the order the validated pipeline uses.

## 2. Input requirements

**CSV** (`--csv FILE`):

| Requirement | Value |
| --- | --- |
| Columns | One per **observed** lead, named from `I II III AVR AVL AVF V1 V2 V3 V4 V5 V6` (case-insensitive, any order). Missing leads are simply absent |
| Sampling rate | **500 Hz** (other rates are not resampled; do not supply them) |
| Units | **millivolts**. Since v2, input reaching ≥ 50 in magnitude is refused as probably not mV (PTB-XL stays far below this); input whose observed leads all stay below 0.01 mV gets a warning |
| Length | at least 4,096 rows (8.192 s); **5,000 rows (10 s) is the validated length**. Longer input is cut to its first 5,000 rows; shorter input runs with a warning |
| Values | numeric, finite (no NaN, no blanks) |
| Observed-lead set | must be one of the **258 L2G lead sets**: any single lead, or 2–7 leads taken only from I, III, V1–V6. Other sets are refused |
| Mains | PTB-XL is 50 Hz mains; the notch is fixed at 50 Hz. 60 Hz recordings are outside the validated setting |
| Data source | **never a PTB-XL fold-10 record** (fold 10 is spent). PTB-XL file names are checked against the frozen split manifest and fold-10 records are refused; a CSV made from a fold-10 record cannot be detected, so do not make one |

**WFDB** (`--wfdb PATH --observed I,III,V1`): a 500 Hz, 12-lead record in PTB-XL lead order and mV. Only the listed leads
are kept, and the others are discarded before anything else is computed.

**What the demo computes internally:**
* **G2.** From the observed raw 500 Hz leads, samples 0–4,095.
* **L2G input.** The observed leads are high-pass and notch filtered, resampled to 250 Hz, stored as float32 (as in the
  validated cache), cut to samples 0–2,047, and min–max scaled per lead.
* **Model input.** 12 masked signal channels plus 12 mask channels.

## 3. How to run

```bash
.venv/Scripts/python.exe demo/infer_partial_ecg.py --csv demo/examples/example_ecg17_I_III_V1_V2_V3_V5.csv demo/example_outputs/my_run
```

```bash
.venv/Scripts/python.exe demo/infer_partial_ecg.py --wfdb C:/PTB-XL/records500/00000/00017_hr --observed I,V2 demo/example_outputs/my_wfdb_run
```

The output folder must not exist; nothing is overwritten. A GPU is used if present; the gate decision itself runs on the
CPU (scikit-learn).

## 4. Output

`OUT_DIR/decision.json` contains:
* the status line, input description and SHA-256, and the samples used;
* the observed leads and any warnings;
* the 30 gate features, and the raw and calibrated Gate B scores;
* the locked threshold (`repr`), and the decision **ACCEPT** (calibrated score ≤ threshold) or **ABSTAIN**;
* the SHA-256 of the protocol, threshold lock, configs, model and calibrator;
* the limitations.

**On ACCEPT only**, `reconstruction_250Hz_minmax.csv` is also written:
* 2,048 rows × 12 leads, at 250 Hz (8.192 s);
* each lead min–max scaled to [0, 1], as the model sees it, so **not in mV**;
* the observed leads are the preprocessed input, unchanged (copy-through).

**On ABSTAIN**, no reconstruction is written, following the abstention policy.

### Fail-closed release and exit codes (version 2, 26 October 2026)

**What changed from v1.** Version 1 reported the copy-through violation count, but did not use it to block release. It is
archived unchanged as `demo/archive/infer_partial_ecg_v1_20261024.py` (`cda635d9…`).

**Release rule.** An accepted reconstruction is released **only if both** of these hold:
* every observed sample is reproduced exactly (copy-through invariant), with no violation reported;
* every value is finite.

Otherwise:
* no reconstruction file is written;
* `decision.json` records `outcome: REFUSED_INVALID_RECONSTRUCTION`, together with the problems found.

**Write order.** The output folder is created only after every computation has succeeded. The reconstruction is written
under a temporary name and then renamed. `decision.json` is written last.

**New fields in `decision.json`:**
* `outcome`: `RELEASED`, `ABSTAINED` or `REFUSED_INVALID_RECONSTRUCTION`;
* `released`: true or false;
* `release_check`: whether it passed, and the problems found;
* `demo_sha256`.

| Exit code | Meaning | Files written |
| --- | --- | --- |
| 0 | RELEASED (gate ACCEPT, release check passed) or ABSTAINED | `decision.json` (+ reconstruction CSV if RELEASED) |
| 1 | Input, configuration or asset refusal, such as a bad header, non-numeric or non-finite values, too few samples, an unsupported lead set, likely not mV, a fold-10 file name, an existing output folder, or a hash mismatch | nothing |
| 3 | REFUSED_INVALID_RECONSTRUCTION (gate ACCEPT, release check failed) | `decision.json` only |

An unexpected error, such as a GPU failure, also exits non-zero and writes nothing.

**Regression tests:** `demo/test_demo_fail_closed.py`, 18 synthetic tests that read no model or PTB-XL data. Run them with:

```bash
.venv/Scripts/python.exe -m pytest demo/test_demo_fail_closed.py -p no:cacheprovider -q
```

**Exact runtime package versions:** `demo/requirements-demo.txt` (includes joblib 1.6.0).

## 5. Verification (development data only)

`demo/verify_demo_on_9b.py` checks the demo against the frozen 9b outputs. 9b is a development role, and fold 10 was not
touched. It used 8 records (64 record-mask rows) and fed the demo only the observed leads of each row.

**Version 2 was checked again on 26 October**, with the same result: **PASS**, 64/64 features, 64/64 bitwise scores, 64/64
E1 outputs on the reconstructions, 0 violations (`demo/verification/verify_9b_2026-10-26_v2.json`). The table below is
the version 1 result (**PASS**, `demo/verification/verify_9b_2026-10-24.json`):

| Check | Result |
| --- | --- |
| G1/G2 features equal to the frozen 9b feature file | 64 / 64 |
| Calibrated Gate B score bitwise equal to the frozen 9b scores | 64 / 64 (so every ACCEPT/ABSTAIN decision is identical) |
| E1 probabilities on the demo's reconstruction equal to the frozen 9b `p_rec` (float32) | 64 / 64, max abs difference 0.0 |
| Copy-through violations | 0 |

**What this shows.** The demo reproduces the validated pipeline exactly, on the same machine and environment. E1 is used
**only** inside this check; the demo never uses E1, labels or D1.

**End-to-end example runs** (`demo/verification/example_runs_console_2026-10-24.log`), on PTB-XL record 17 (role 9b):

| Input | Decision | Calibrated score | Threshold |
| --- | --- | --- | --- |
| `examples/example_ecg17_I_III_V1_V2_V3_V5.csv` | ACCEPT (reconstruction written) | 0.0413 | 0.0593 |
| `examples/example_ecg17_V6_only.csv` | ABSTAIN (no reconstruction) | 0.1229 | 0.0593 |

A fold-10 PTB-XL file name was refused, as intended.

## 6. Limitations

* **Unvalidated outside PTB-XL.** There is no validation on other devices, populations, sampling rates, 60 Hz mains, noise
  types or partially missing leads.
* **Whole-lead missingness only.** The gate and the reconstructor were evaluated only with whole leads missing, on the
  258 lead sets.
* **ACCEPT is a deferral decision, not a diagnosis.** On fold 10, 1.7 % of accepted rows still changed E1's BBB decision,
  and the gate released only 7.9 % of E1-called CLBBB rows.
* **No clinical use.** The reconstruction is scaled per lead to [0, 1]. It is a research output and must not be used for
  clinical decisions.
* **Bitwise reproducibility is environment-bound.** Results reproduce bit for bit only in the pinned environment (Windows
  11, Python 3.12.10, the library versions in the threshold lock); on other hardware the reconstruction may differ in the
  last bits.

The example files contain data derived from PTB-XL v1.0.3 (PhysioNet, CC BY 4.0; Wagner et al., *Scientific Data* 7,
154, 2020).
