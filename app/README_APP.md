# ECG capstone research-demo application

> **Research prototype only — not a medical device. Do not use for diagnosis or treatment.**

> **Public-release note (28 Oct 2026).** This file is the project's internal app record, kept for provenance. In this
> repository:
> * the PTB-XL record-17 "development fixtures" and `test_real_pipeline.py` are **not distributed**: no patient recording is
>   published. The built-in examples are now the synthetic files in `examples/synthetic/`;
> * the frontend runs in a separate environment (see `app/requirements-frontend.txt`);
> * `.claude/launch.json` is not included.
>
> For setup and the demo, follow the top-level `README.md` and `SUPERVISOR_DEMO_GUIDE.md`.

This local application demonstrates the project's frozen selective-release pipeline on one partial 12-lead ECG at a time.

**The pipeline:**
1. G1 mask-geometry and G2 observed-lead quality features feed **Gate B**.
2. Platt calibration (fitted on 9a) and the threshold locked on 9b give **ACCEPT** or **ABSTAIN**.
3. **Only after ACCEPT**, the L2G-based reconstructor **G v0.1** fills in the missing leads, with the observed samples
   copied through exactly.
4. The reconstruction is released only if a fail-closed release check passes.

**What it is not.** It is not production software, and it is not clinically validated. The gate was evaluated once, on
PTB-XL fold 10, with synthetic whole-lead missingness. It has no external validation, and it withholds most
bundle-branch-block (BBB) ECGs; see report v1.4, Sections 8–10.

## 1. Design: two processes, two environments

| Part | Runs in | What it does |
| --- | --- | --- |
| Backend: `app/backend/` | The project's pinned `.venv` (Python 3.12.10, torch 2.11 + CUDA, scikit-learn 1.9.0) | A local JSON API on `127.0.0.1:8765`. It reuses the stage 10 demo functions (`demo/infer_partial_ecg.py`) unchanged, through `app/backend/pipeline.py` |
| Frontend: `app/frontend/` | The **system** Python 3.12 (Streamlit 1.58 is installed there) | The user interface. It holds no model code and only calls the backend over HTTP |

**Nothing was installed into `.venv`.** The research pipeline's bit-for-bit reproduction depends on its exact package
versions. No dependency file (`requirements*.txt`) was changed, and no research code (`src/`, `configs/`, `tests/`) was
touched.

**Safeguards built in:**
* **At start-up,** the backend checks the SHA-256 of the following, and refuses to start if any differs:
  * G v0.1 (`d515875a…`);
  * Gate B (`7b863410…`);
  * the calibrator (`341ba86a…`);
  * the threshold lock (`bc69c301…`, t = 0.05927309439311807);
  * protocol v0.2 (`38ad0a86…`);
  * the reused demo code (`62d05bc7…`).
* **Local only.** The backend binds to `127.0.0.1` and refuses any other address.
* **Strict request schema.** Unknown fields, such as a complete ECG, labels or D1, are refused. Only observed leads ever
  reach the gate.
* **Nothing is stored.** Uploads are processed in memory. Nothing is written to disk, and request bodies or waveform values
  are never logged (the log holds the method, path and status only).
* **ABSTAIN means no reconstruction.** The reconstructor is never called after ABSTAIN.
* **Fail closed.** If the release check fails, nothing is released.

## 2. Start the application (Windows PowerShell, from the repository root)

```powershell
cd "<path to your clone of this repository>"
```

**Terminal 1, the backend (pinned environment).** It loads the frozen models at start-up, which takes a few seconds.

```powershell
.\.venv\Scripts\python.exe -m app.backend.server
```

Wait for `ECG research-demo backend on http://127.0.0.1:8765 (assets verified; Ctrl+C to stop)`. Optional flags:
`--port 8765`, and `--no-preload` to load the models on first use.

**Terminal 2, the frontend (system Python).**

```powershell
python -m streamlit run app\frontend\streamlit_app.py
```

Streamlit opens `http://localhost:8501`. The sidebar should show **"Connected · frozen assets verified"**, together with
the model hashes and the threshold. Stop either process with Ctrl+C.

**Alternative.** `.claude/launch.json` defines both servers (`ecg-backend`, `ecg-frontend`) for the Claude Code browser
pane.

## 3. Input format (schema `partial-ecg-csv/1`)

* **A CSV file.** One column per **observed** lead. The header uses the lead names `I, II, III, AVR, AVL, AVF, V1, V2, V3,
  V4, V5, V6` (any order; case-insensitive). Missing leads are simply absent.
* **500 Hz, millivolts.** At least 4,096 rows (8.192 s); 5,000 rows (10 s) is the validated length, and longer input is cut
  to its first 5,000 rows.
* **Supported lead sets.** The observed set must be one of the **258** L2G lead sets: any single lead, or 2–7 leads from
  I, III, V1–V6.
* **Values.** Finite numbers, below 50 mV in magnitude; for example, microvolt input is refused.
* **Rectangular.** Every row must have exactly as many fields as the header. See the known gap in Section 7.
* **Size.** At most 3,000,000 characters and 60,000 rows.
* **No fold-10 files.** A PTB-XL file name from fold 10 is refused. Do not use fold 10, which is spent.

**In the interface**, you can choose:
* **Two development fixtures**, from PTB-XL record 17 (role 9b, never fold 10; made by `demo/make_example_inputs.py`). Each
  file holds only its observed leads:
  * the 6-lead file (I, III, V1, V2, V3, V5);
  * the V6-only file.
* **A synthetic signal**, clearly labelled as not an ECG; the gate's decision on it means nothing.
* **Your own CSV upload.**

The **observed-leads** box is the whole-lead mask: deselecting a lead treats it as missing, and its column is not sent.

**Calling the API directly.** `POST /api/infer` with the JSON object
`{"schema_version": "partial-ecg-csv/1", "sampling_rate_hz": 500, "units": "mV", "observed_leads": [...], "csv": "...",
"source_name": "optional.csv"}`. `GET /api/health` and `GET /api/info` report the status, versions, hashes and the 258
lead sets.

## 4. Expected output

| Outcome | Meaning | What is shown |
| --- | --- | --- |
| `RELEASED` | Calibrated score ≤ t, so the gate accepted, and the release check passed | Score against threshold, and the 12-lead reconstruction: 250 Hz, 2,048 samples, each lead min–max scaled to [0, 1] (the model's scale, **not mV**). Observed leads in navy are copied exactly; reconstructed leads are in green |
| `ABSTAINED` | Calibrated score > t | Score against threshold. **No reconstruction is computed or shown** |
| `REFUSED_INVALID_RECONSTRUCTION` | The gate accepted, but the release check failed (copy-through, finite values or shape) | Nothing is released |
| Error 422 `INVALID_*` / `UNSUPPORTED_*` / `FORBIDDEN_OR_UNKNOWN_FIELD` / `FOLD10_REFUSED` | The input was refused before any computation | The reason |

**Reference results** (the same as the stage 10 command-line demo):
* **6-lead fixture:** ACCEPT, score 0.0413 (0.04132697328467846), RELEASED.
* **V6-only fixture:** ABSTAIN, score 0.1229 (0.12286176065177754).

## 5. Tests

**Backend** (pinned `.venv`): `app/tests/backend`, 42 tests.

```powershell
.\.venv\Scripts\python.exe -m pytest app/tests/backend -q
```

* **40 run against stand-ins** for the gate and the reconstructor (mocked). They cover:
  * the schema, file types, NaN or infinite values, masks, shapes and the ragged-CSV guard;
  * the startup integrity check;
  * ACCEPT leads to reconstruction; ABSTAIN leads to none;
  * a failed release check refuses output;
  * exact copy-through;
  * the gate's inputs are limited to the 30 whitelisted features;
  * no file writes;
  * the HTTP status codes, local-only binding, and no body logging.
* **2 run the real frozen pipeline** (`test_real_pipeline.py`). On the two fixtures they reproduce the stage 10 demo's
  gate scores exactly and its reconstruction bit for bit; ABSTAIN never calls the reconstructor.

**Frontend** (system Python): `app/tests/frontend`. 19 tests run against a stand-in backend (mocked). They cover start-up,
the warning, backend down, ABSTAIN shows no reconstruction, RELEASED shows it, mask selection, stale results, refusal, reset
and the upload checks. One more test is a real end-to-end run that needs the backend running.

```powershell
python -m pytest app/tests/frontend -q
$env:ECG_APP_E2E_URL = "http://127.0.0.1:8765"; python -m pytest app/tests/frontend -k e2e -q
```

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `REFUSING TO START: frozen asset check failed` | A frozen model, calibrator, lock, protocol or the demo code differs from its pinned hash. Do not run; restore the original file |
| The sidebar says "Backend not available" | Start the backend (Terminal 1) and check that the URL in the sidebar is `http://127.0.0.1:8765` |
| `OSError: [WinError 10048]` (port in use) | Another process uses the port: stop it, or start with `--port 8766` and change the sidebar URL |
| `ModuleNotFoundError: streamlit` | Run the frontend with the system Python (`python`), not `.venv` |
| `ModuleNotFoundError: torch` or `sklearn` | Run the backend with `.\.venv\Scripts\python.exe` |
| "not one of the 258 L2G lead sets" | Choose a single lead, or 2–7 leads from I, III, V1–V6 |
| "row N has … fields but the header has …" | The CSV is not rectangular; fix the file (see Section 7) |
| "probably not in mV" | Convert microvolts to millivolts |
| The first request is slow | The models load on first use only with `--no-preload` |

## 7. Limitations and known gaps

* **Evaluation.** One pre-registered evaluation on PTB-XL fold 10 (now spent). There is no external or clinical
  validation, and ACCEPT is not evidence of diagnostic fidelity.
* **CLBBB.** Gate B releases only 7.9 % of E1-called CLBBB rows. It is a deferral rule, not a safety claim.
* **D1.** The gate's target is disagreement with one frozen classifier, not clinical harm or truth.
* **Format.** Only the documented CSV format is supported; other ECG file formats are not accepted.
* **Known gap in the stage 10 command-line demo.** `demo/infer_partial_ecg.py --csv` reads files with pandas. If every
  data row has one field more than the header, pandas silently treats the first column as an index, and every lead shifts
  by one column. This application refuses such files, in both the frontend and the backend. The command-line demo itself
  is a pinned stage 10 deliverable and was not changed.
* **L2G v0.2-exp** (exploratory, development data only) is not used anywhere in this application.
