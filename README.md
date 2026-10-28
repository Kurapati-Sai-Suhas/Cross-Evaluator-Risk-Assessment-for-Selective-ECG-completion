# Cross-Evaluator Risk Assessment for Selective ECG Completion

**Capstone project:** *Risk-Controlled Selective ECG Completion for Diagnostic Distortion Prevention* (BTech CSE (AI&ML),
2026–2027). The author list is to be confirmed by the team.

> **Research prototype only — not a medical device. Do not use for diagnosis or treatment.** The system has no clinical,
> reader or independent external validation.

## What it does

The input is a 12-lead ECG in which some leads are missing. For each such input, the system decides **before
reconstructing** whether a reconstruction of the missing leads should be released:

1. It receives the observed leads and the lead mask (which leads are present).
2. It computes 15 **G1** mask-geometry features and 15 **G2** observed-signal-quality features, from the observed leads only.
3. The frozen **Gate B** Random Forest scores the input. **Platt calibration** (fitted on role 9a) turns the score into a
   risk. The **locked threshold** t = 0.05927309439311807 (fixed on role 9b) gives **ACCEPT** (risk ≤ t) or **ABSTAIN**.
4. **Only after ACCEPT**, the frozen L2G-ECG-based reconstructor **G v0.1** fills in the missing leads. The observed samples
   are copied through exactly.
5. A fail-closed **release check** must pass before anything is released: shape, copy-through, observed samples
   bit-identical, and all values finite.
6. **After ABSTAIN**, no reconstruction is computed or released. ABSTAIN is a deferral, not a diagnosis.

**What the gate predicts.** The gate was trained to predict an evaluator-relative target, D1: whether a frozen classifier
(E1) changes its complete-LBBB or complete-RBBB decision between the complete ECG and the reconstruction.

**Where E1 is used.** Offline only, to create that target. It is **not** part of this runtime.

## What the evidence supports, and what it does not

**Result.** One pre-registered, hash-locked evaluation was run, on PTB-XL fold 10 (now spent). At its locked threshold,
Gate B released 80.1 % of reconstructions:
* **D1 among the released ones:** 0.0171, against 0.0415 when everything is accepted.
* **H1 (against accept-all):** Δ = −0.0244, 95 % interval [−0.0293, −0.0197].
* **H2 (against mask geometry alone at 80 % coverage):** Δ = −0.0149 [−0.0192, −0.0111].
* **Decision:** both null hypotheses were rejected under the locked rule.

**These limits must accompany that result:**
* **D1 is not truth.** It is disagreement with one frozen classifier, not diagnostic truth or clinical harm.
* **Deferral drives the gain.** Much of the pooled gain comes from withholding bundle-branch-block ECGs: only 7.9 % of
  E1-called CLBBB rows were released.
* **Simpler comparators.** Gate B was not shown to beat E1's own margin (M2) or quality features alone (M3).
* **The external check is exploratory.** On CPSC 2018, coverage fell to 0.582, and a conformal variant missed its target.
  This is not independent confirmation.
* **Morphology.** QRS differences were smaller than the delineator's own error.
* **Novelty is unknown.** No systematic prior-art search has been done yet.

## Repository layout

| Path | Contents |
| --- | --- |
| `app/` | Local application: `backend/` (JSON API on 127.0.0.1:8765) and `frontend/` (Streamlit UI); `tests/` |
| `demo/infer_partial_ecg.py` | Command-line inference for one CSV (the functions the app reuses) |
| `src/`, `configs/`, `tests/` | Research code and configuration, byte-identical to the frozen project (code-tree SHA-256 `7cb6c7f9…`, 93 files) |
| `artifacts/` | **Frozen runtime models and fitted components.** See `artifacts/runtime/manifest.json` |
| `examples/synthetic/` | Generated synthetic inputs (**not ECG recordings**) and their screening log |
| `tools/verify_runtime_artifacts.py` | Checks the frozen artifacts against the manifest, and optionally loads them |
| `SUPERVISOR_DEMO_GUIDE.md`, `SUPERVISOR_DEMO_CHEATSHEET.md` | Step-by-step demonstration |

## Setup on Windows (VS Code + PowerShell)

**Requirements:**
* Windows 10 or 11.
* **Python 3.12** (tested with 3.12.10).
* About 6 GB of disk space for the environments.
* An NVIDIA GPU is optional: the PyTorch CUDA build also runs on the CPU.

Open the cloned folder in VS Code, then open a PowerShell terminal (**Terminal → New Terminal**).

```powershell
git clone https://github.com/Kurapati-Sai-Suhas/Cross-Evaluator-Risk-Assessment-for-Selective-ECG-completion.git
cd Cross-Evaluator-Risk-Assessment-for-Selective-ECG-completion
```

**Backend environment** (pinned research versions):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r demo\requirements-demo.txt
```

**Frontend environment** (kept separate, so the backend environment is never changed):

```powershell
python -m venv .venv-frontend
.\.venv-frontend\Scripts\python.exe -m pip install -r app\requirements-frontend.txt
```

In VS Code, choose **Python: Select Interpreter → `.venv\Scripts\python.exe`**.

**No activation is needed:** every command here calls the interpreter by its path.

## Check the frozen models (no training)

```powershell
.\.venv\Scripts\python.exe tools\verify_runtime_artifacts.py --load
```

Expected last line: `PASS: 14 runtime artifacts match the manifest and load for inference`.

## Run

**Command line** (writes `decision.json`, plus a reconstruction CSV only on a release; the output folder must not exist):

```powershell
.\.venv\Scripts\python.exe demo\infer_partial_ecg.py --csv examples\synthetic\synthetic_cand01.csv demo_outputs\cand01
.\.venv\Scripts\python.exe demo\infer_partial_ecg.py --csv examples\synthetic\synthetic_cand04.csv demo_outputs\cand04
```

Expected:
* `cand01`: ACCEPT, calibrated score 0.0026, RELEASED.
* `cand04`: ABSTAIN, score 0.0762; no reconstruction file is written.

**Application.** Use two terminals:

```powershell
# Terminal 1: backend (verifies every frozen asset's SHA-256 at start-up, then loads the models)
.\.venv\Scripts\python.exe -m app.backend.server
```

```powershell
# Terminal 2: frontend, bound to this computer only
.\.venv-frontend\Scripts\python.exe -m streamlit run app\frontend\streamlit_app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Stop each process with **Ctrl+C**.

## Tests (safe set; no data needed)

```powershell
.\.venv\Scripts\python.exe -m pytest app\tests\backend demo\test_demo_fail_closed.py -q
.\.venv-frontend\Scripts\python.exe -m pytest app\tests\frontend -q
```

**What they cover.** The backend set includes `test_synthetic_release.py`, which runs the **real** frozen pipeline:
* cand01 must be ACCEPTED with exact copy-through;
* cand04 must ABSTAIN without calling the reconstructor.

**Do not run** `tests/test_protocol_v0_2.py` or `tests/test_fold10_pathway.py`. They belong to the project's protected
final-evaluation pathway. Several other tests in `tests/` need the PTB-XL data and project caches, which are not
distributed, so they skip or fail in a fresh clone.

## Why nothing is retrained

The application and the demo only **load** the files listed in `artifacts/runtime/manifest.json`:
* **Integrity first.** The backend verifies their SHA-256 at start-up and refuses to start on any mismatch.
* **Inference only.** The reconstructor runs under `torch.inference_mode()` and must be in eval mode.
* **No training is run.** `src/eval/generate_d1.py` imports two preprocessing helpers from `src/train/train_l2g.py`, so
  that module is loaded. Its `train()` function, the only place optimisers are created, is never called.

**Checked on 28 Oct 2026, after a real ACCEPT request through the app service:**
* no optimiser exists;
* the generator is not in training mode;
* no parameter has a gradient.

## What is not in this repository

* **Datasets and patient recordings.** No PTB-XL or CPSC 2018 recordings and no patient-level metadata.
* **Research evidence.** No D1 tables or gate features, the E1 classifier, the fold-10 or CPSC outputs, authorisation
  records, or the project's internal reports.
* **Real-record examples.** The development examples derived from a PTB-XL record are replaced by the synthetic examples.

See `NOTICE.md` for data attribution (PTB-XL, CC BY 4.0), the architecture citation (L2G-ECG), and licence status.
