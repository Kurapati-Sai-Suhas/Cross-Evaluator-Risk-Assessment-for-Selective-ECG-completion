# Supervisor demo guide

**Project:** Risk-Controlled Selective ECG Completion for Diagnostic Distortion Prevention (repository: Cross-Evaluator Risk
Assessment for Selective ECG Completion).

> **Research prototype only — not a medical device. Do not use for diagnosis or treatment.** Every input used in this demo
> is a **generated synthetic signal**, not an ECG recording and not patient data. The demo shows that the software works.
> It is **not** clinical evidence.

**How this guide was checked.** Every command and expected output below was run on 28 Oct 2026 from a fresh clone of this
repository on Windows 11 (Python 3.12.10). The outputs quoted in the "You should see" lines are the ones actually
observed.

---

## A. Preparation checklist (do this before the meeting)

| Check | How | Pass when |
| --- | --- | --- |
| 1. Right folder and version | Steps B1 and B3 | `git log -1` shows the commit you rehearsed, and `git status --short` is empty |
| 2. Python and environments | Step B2 | Python 3.12.x; `.venv` and `.venv-frontend` exist |
| 3. Dependencies | Step B3 | torch 2.11.0, scikit-learn 1.9.0 and streamlit 1.58.0 are printed |
| 4. Frozen models present and unchanged | Step B4 | `PASS: 14 runtime artifacts match the manifest and load for inference` |
| 5. Backend and frontend start | Steps B5–B6 | the sidebar shows **Connected · frozen assets verified** |
| 6. Synthetic inputs present | `dir examples\synthetic` | `synthetic_cand01.csv` and `synthetic_cand04.csv` are listed |
| 7. No training or protected evaluation | (by design) | You run only the commands in this guide. None of them trains, fits, calibrates or opens evaluation data |
| 8. Warnings visible | Step B6 | the red banner reads "Research prototype only — not a medical device…" |

**Things to have open:** two PowerShell terminals in VS Code, and a browser.

---

## B. Exact Windows PowerShell commands

### B1. Open the project folder

Open the folder of your clone in VS Code (**File → Open Folder**). Then open a terminal (**Terminal → New Terminal**). If
you cloned into `ecg-release` under your user folder, as when this guide was checked:

```powershell
cd "$HOME\ecg-release\Cross-Evaluator-Risk-Assessment-for-Selective-ECG-completion"
git log -1 --oneline
git status --short
```

### B2. Python environment

This guide calls each interpreter by its path, so **no activation is needed**:
* `.venv` is the backend;
* `.venv-frontend` is the user interface.

If they do not exist yet, create them once (this takes a while: PyTorch is large):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r demo\requirements-demo.txt
python -m venv .venv-frontend
.\.venv-frontend\Scripts\python.exe -m pip install -r app\requirements-frontend.txt
```

### B3. Check Python and package versions

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch, sklearn; print('torch', torch.__version__, '| sklearn', sklearn.__version__, '| CUDA', torch.cuda.is_available())"
.\.venv-frontend\Scripts\python.exe -c "import streamlit; print('streamlit', streamlit.__version__)"
```

You should see `Python 3.12.x`, `torch 2.11.0+cu128 | sklearn 1.9.0`, and `streamlit 1.58.0`. CUDA may be True or False;
both work.

### B4. Verify the frozen models and checksums

```powershell
.\.venv\Scripts\python.exe tools\verify_runtime_artifacts.py --load
```

You should see 14 `OK` lines, then:
* `LOAD gate: RandomForestClassifier with 30 features; calibrator LogisticRegression; threshold 0.05927309439311807`;
* `LOAD reconstructor: ECGen, eval mode = True, device = …`;
* `PASS: 14 runtime artifacts match the manifest and load for inference`.

**Point out:** the models are only **loaded and checked**. Nothing is trained.

### B5. Start the backend (Terminal 1)

```powershell
.\.venv\Scripts\python.exe -m app.backend.server
```

**What happens:**
* Wait for: `ECG research-demo backend on http://127.0.0.1:8765 (assets verified; Ctrl+C to stop)`.
* It checks the SHA-256 of every frozen asset **before** loading, and refuses to start if any differs.
* It listens only on 127.0.0.1.

### B6. Start the frontend (Terminal 2) and open the app

```powershell
.\.venv-frontend\Scripts\python.exe -m streamlit run app\frontend\streamlit_app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**.

**You should see:**
* the title **Risk-Controlled Selective ECG Completion**;
* the red banner **"Research prototype only — not a medical device. Do not use for diagnosis or treatment."**;
* in the sidebar, **Connected · frozen assets verified**, with the hashes of G v0.1 (`d515875a7783…`), Gate B
  (`7b863410bd34…`) and the calibrator (`341ba86a3e2b…`), and **Threshold t: 0.05927309439311807 (locked on 9b)**.

### B7. Run the synthetic demo

**In the app** (Section C), or **from the command line** in a third terminal:

```powershell
.\.venv\Scripts\python.exe demo\infer_partial_ecg.py --csv examples\synthetic\synthetic_cand01.csv demo_outputs\cand01
.\.venv\Scripts\python.exe demo\infer_partial_ecg.py --csv examples\synthetic\synthetic_cand04.csv demo_outputs\cand04
```

The output folder must not exist yet. Use a new name, such as `demo_outputs\cand01_b`, if you repeat the run.

### B8. Run the safe readiness tests

```powershell
.\.venv\Scripts\python.exe -m pytest app\tests\backend demo\test_demo_fail_closed.py -q
.\.venv-frontend\Scripts\python.exe -m pytest app\tests\frontend -q
```

When this guide was checked:
* **Backend set:** 65 passed. It includes the real-model test on cand01 and cand04.
* **Frontend set:** 19 passed; 1 skipped (the optional live test).

**Never run** `tests\test_protocol_v0_2.py` or `tests\test_fold10_pathway.py`. They belong to the protected
final-evaluation pathway.

### B9. Stop everything cleanly

Press **Ctrl+C** in Terminal 2 (frontend), then in Terminal 1 (backend). Nothing needs cleaning up: the app stores no
uploads.

---

## C. Demonstrating both decisions

These two examples were screened in a fixed order with no selection on output quality; see
`examples/synthetic/screening_log.json`. Both decisions were re-verified on 28 Oct 2026, in the app and on the command
line.

### C1. ACCEPT: synthetic example cand01 (observed leads I, III, V1, V2, V4, V6)

1. **Select the input.** Under **Input**, choose **"Synthetic example cand01: 6 observed leads (generated; NOT an ECG
   recording)"**. This is the default.
2. **Show the input.** The grey caption says the file is a synthetic signal, not an ECG recording. The plot is titled
   **Observed ECG (input)**, with "observed: I, III, V1, V2, V4, V6 · missing: II, AVR, AVL, AVF, V3, V5".
3. **Run it.** Click **Run the gate**.
4. **You should see:**
   * the green **"ACCEPT · the reconstruction is released"**;
   * **Calibrated risk score 0.0026**, **Locked threshold t 0.0593**, **Gate decision ACCEPT**;
   * the section **Reconstruction (released)**;
   * the note "ACCEPT is not evidence of diagnostic fidelity…".
5. **Point out:**
   * the score is **at or below** t, so the reconstruction was computed **only then**;
   * the release check passed: the observed samples were reproduced exactly, and all values are finite. Open
     **Technical details** to show this.

**On the command line**, the run prints `"outcome": "RELEASED"`, `"gate_decision": "ACCEPT"` and
`"calibrated_score": 0.0025646914922358727`. It writes `demo_outputs\cand01\decision.json` and
`reconstruction_250Hz_minmax.csv`: 2,048 rows × 12 leads, 250 Hz, each lead scaled to [0, 1].

### C2. ABSTAIN: synthetic example cand04 (only lead V2 observed)

1. **Select the input.** Choose **"Synthetic example cand04: lead V2 only (generated; NOT an ECG recording)"**. The input
   line reads "observed: V2 · missing: I, II, III, …".
2. **Run it.** Click **Run the gate**.
3. **You should see:**
   * the amber **"ABSTAIN · no reconstruction was computed"**;
   * **Calibrated risk score 0.0762**, which is **above** t = 0.0593;
   * **Gate decision ABSTAIN**;
   * **"No reconstruction is shown for this input."**
4. **Point out:**
   * the reconstructor was **never called**: the test `test_cand04_abstains_and_never_reconstructs` proves this;
   * ABSTAIN means "do not rely on a reconstruction for this input". It is **not** a diagnosis.

**On the command line**, the run prints `"outcome": "ABSTAINED"`, `"gate_decision": "ABSTAIN"` and
`"calibrated_score": 0.07616321041813778`. It writes only `demo_outputs\cand04\decision.json`; **no** reconstruction file
is written.

**Optional.** The **Reset** button clears the result. Changing the observed-lead selection also clears a stale result.

---

## D. What to say (plain-English script, about 3 minutes)

1. **The problem.** "Many devices record only some of the 12 ECG leads. Reconstruction models can estimate the missing
   leads, and they are usually judged by how similar the waveforms look."
2. **Why similarity is not enough.** "A reconstruction can look almost identical and still change what a diagnostic
   classifier decides. For bundle branch block, a small change in QRS shape can flip the decision."
3. **Our idea.** "Before reconstructing, a gate decides whether a reconstruction should be released for this particular
   input."
4. **What the gate sees.** "Only information available at use time: which leads are present (mask geometry) and the
   quality of those observed leads. It never sees the complete ECG, the labels, the classifier's output or the true error."
5. **Why reconstruct only after ACCEPT.** "So that the decision cannot depend on the reconstruction itself, and nothing
   is produced when we abstain. After ACCEPT, the observed samples are copied through exactly, and a release check must
   pass."
6. **What ABSTAIN means.** "It is a deferral: 'don't rely on a reconstruction here'. It says nothing about whether the
   patient has a disease."
7. **What the results support.** "In one pre-registered evaluation on a held-out PTB-XL fold, released reconstructions
   changed our frozen classifier's BBB decision less often than accepting everything: 1.71 % against 4.15 %. That was at
   about 80 % coverage, and both pre-registered null hypotheses were rejected."
8. **What they do not show.**
   * "The target is disagreement with one frozen classifier, not diagnostic truth or clinical harm."
   * "Much of the gain comes from withholding BBB ECGs; only 7.9 % of complete-LBBB cases were released."
   * "Gate B was not shown to beat simpler comparators."
   * "The CPSC 2018 check was exploratory, and coverage dropped to 0.58."
   * "There is no independent evaluator or clinical validation yet, and novelty is not established."
9. **Safety.** "This is a research prototype, not a medical device. Today's inputs are generated signals, not patient
   recordings."

---

## E. Troubleshooting and fallback

| Problem | What to do |
| --- | --- |
| `verify_runtime_artifacts.py` says MISSING or SHA-256 differs | Do not demo inference. Re-clone the repository, or run `git status` and `git checkout -- artifacts`. Never replace a model file with a different one |
| `No module named torch` (or sklearn) | You used the wrong interpreter. Use `.\.venv\Scripts\python.exe …`. In VS Code, select `.venv\Scripts\python.exe` |
| `No module named streamlit` | Run the frontend with `.\.venv-frontend\Scripts\python.exe` |
| `pip install` fails | Check the internet connection and that you are on Python 3.12, then rerun the B2 commands |
| Backend refuses to start: `IntegrityError` | A frozen file changed or is missing. Run B4 to see which; fix it by re-cloning |
| Port 8765 or 8501 already in use | Stop the old process (Ctrl+C in its terminal), or run `Get-NetTCPConnection -LocalPort 8765 -State Listen` to find it. Or use other ports: backend `--port 8766`, then type `http://127.0.0.1:8766` into the sidebar's **Local backend URL**; frontend `--server.port 8502` |
| Sidebar says "Backend not available" | Start the backend first (B5) and wait for its ready line, then reload the page |
| "Input refused … CSV" | The CSV needs one column per observed lead named `I, II, III, AVR, AVL, AVF, V1–V6`, 500 Hz, in mV, with at least 4,096 rows |
| "not one of the 258 L2G lead sets" | Only a single lead, or 2–7 leads from I, III, V1–V6, are supported. Choose a supported set |
| "probably not in mV; refusing" | Values must be in millivolts (below 50 in magnitude) |
| Command-line run says the output folder exists | Use a new output folder name |
| App fails to start at all | Fall back to the command line (B7), which uses the same frozen functions |

**If the model files are unavailable or fail verification:**
* Do **not** show inference results.
* You may show the verification output itself.
* You may show the mocked frontend tests (`pytest app\tests\frontend -q`): they exercise the interface with a stand-in
  backend. Say clearly that this is a **software-only check, not inference**.
* You may show the recorded screening log (`examples\synthetic\screening_log.json`), saying it is a **recorded** result,
  not a live one.
* Never present a result you did not just produce.

---

## F. Rehearsal checklist (from a fresh terminal)

- [ ] **Folder and version:** `cd` into the clone; `git log -1 --oneline` matches; `git status --short` is empty.
- [ ] **Artifacts:** `.\.venv\Scripts\python.exe tools\verify_runtime_artifacts.py --load` ends with **PASS**.
- [ ] **Backend:** Terminal 1 shows "assets verified".
- [ ] **Frontend:** Terminal 2 starts; http://127.0.0.1:8501 shows the red warning and "Connected · frozen assets verified".
- [ ] **ACCEPT:** cand01 → ACCEPT, 0.0026, reconstruction shown.
- [ ] **ABSTAIN:** cand04 → ABSTAIN, 0.0762, "No reconstruction is shown".
- [ ] **Command line:** cand01 → RELEASED; cand04 → ABSTAINED, with no reconstruction file.
- [ ] **Tests:** the safe set passes (65 backend, 19 frontend).
- [ ] **Shutdown:** Ctrl+C in both terminals.
- [ ] **Script:** say Section D aloud once, including what the results do **not** show.
