# Runtime artifacts (frozen; inference only)

`manifest.json` lists **every non-code file the inference service opens**: 14 files, found by tracing file opens during a
real ACCEPT and a real ABSTAIN request on 28 Oct 2026. For each file it gives:
* path, role and component;
* size and SHA-256;
* expected input and dependencies;
* provenance and load-verification result.

`checksums.sha256` holds the same hashes in `sha256sum` format.

**The files stay at the paths the code reads them from.** They were not moved, so the application's own start-up
integrity check (`app/backend/integrity.py`) and the hash-checked loaders (`src/gate/m4b1.py`) work unchanged.

## Check them

```powershell
python tools\verify_runtime_artifacts.py
.\.venv\Scripts\python.exe tools\verify_runtime_artifacts.py --load
```

* The first command checks sizes and hashes. It needs only Python.
* The second also loads the gate, the calibrator, the threshold and the reconstructor through the application's own
  start-up path, with the reconstructor in eval mode.

## What is used for the decision

| File | Use |
| --- | --- |
| `artifacts/gate/v0_2/m4b1/01_fit/models/gate_B.joblib` | Gate B Random Forest (30 features: G1 + G2) |
| `artifacts/gate/v0_2/m4b1/02_calibration/calibrator_gate_B.joblib` | Platt calibration (role 9a) |
| `artifacts/locks/threshold_lock_v0_2.json` | locked threshold t = 0.05927309439311807 (role 9b); ACCEPT iff score ≤ t |
| `artifacts/runs/l2g_v0_1_run02/best.pt` | reconstructor G v0.1, called **only after ACCEPT** |

**Loaded but not used for the decision.** The Gate A and M3 models and the A, M2 and M3 calibrators are loaded because
the hash-checked loaders load every fitted method recorded in the stage summaries.

**About `best.pt`.** It also stores training-time optimiser and discriminator state. Only its `generator` weights are
loaded. The file is distributed unchanged, because the application pins it by SHA-256 (`d515875a…`).

## Not included

| Item | Why |
| --- | --- |
| E1 classifier | offline target generation and evaluation only; never on the runtime path |
| L2G v0.2-exp | exploratory; not part of the app |
| Datasets, D1 tables, gate features, split manifest | patient-level research data; not distributed |
| Fold-10 and CPSC outputs, authorisation records | protected research evidence; never distributed |

## Size

`best.pt` is 64,019,691 bytes. That is below GitHub's 100 MB per-file limit, so it is stored with ordinary Git, not Git
LFS. GitHub shows a warning for files over 50 MB, which is expected.
