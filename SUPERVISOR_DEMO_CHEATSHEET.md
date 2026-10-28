# Demo cheat sheet

The full guide, `SUPERVISOR_DEMO_GUIDE.md`, is authoritative.

> **Research prototype only — not a medical device.** The inputs are synthetic, not ECG recordings.

**1. Check** (from the project folder):

```powershell
.\.venv\Scripts\python.exe tools\verify_runtime_artifacts.py --load
```

Expected: `PASS: 14 runtime artifacts match the manifest and load for inference`.

**2. Terminal 1, the backend:**

```powershell
.\.venv\Scripts\python.exe -m app.backend.server
```

Wait for "assets verified".

**3. Terminal 2, the frontend:**

```powershell
.\.venv-frontend\Scripts\python.exe -m streamlit run app\frontend\streamlit_app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501.

**4. Show both decisions:**

| Input | Expected |
| --- | --- |
| **Synthetic example cand01** (I, III, V1, V2, V4, V6) | ACCEPT · score 0.0026 ≤ t 0.0593 · reconstruction released |
| **Synthetic example cand04** (V2 only) | ABSTAIN · score 0.0762 > t 0.0593 · no reconstruction computed |

**5. Command-line fallback:**

```powershell
.\.venv\Scripts\python.exe demo\infer_partial_ecg.py --csv examples\synthetic\synthetic_cand01.csv demo_outputs\c01
.\.venv\Scripts\python.exe demo\infer_partial_ecg.py --csv examples\synthetic\synthetic_cand04.csv demo_outputs\c04
```

**6. Stop:** Ctrl+C in Terminal 2, then in Terminal 1.

**One-minute explanation.** A gate looks only at which leads are present and how clean they are. It then decides whether a
reconstruction of the missing leads may be released.
* **ACCEPT:** the frozen L2G-based model reconstructs, copies the observed samples exactly, and must pass a release check.
* **ABSTAIN:** nothing is reconstructed. That is a deferral, not a diagnosis.

**Supported** (one pre-registered PTB-XL test): fewer evaluator-relative BBB decision changes among released
reconstructions, 1.71 % against 4.15 %, at about 80 % coverage.

**Not shown:**
* clinical benefit or diagnostic truth;
* a gain over simpler comparators;
* independent external confirmation;
* novelty.

Also note that CLBBB ECGs are mostly withheld (7.9 % released).
