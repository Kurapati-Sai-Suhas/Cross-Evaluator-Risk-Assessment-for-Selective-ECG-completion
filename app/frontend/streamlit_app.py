"""Streamlit frontend for the ECG capstone research demo (stage 14). It holds no model code: every decision comes from the
local backend (app/backend/server.py), which runs the frozen pipeline in the project's pinned environment.

    python -m streamlit run app/frontend/streamlit_app.py          (system Python; start the backend first)

Research prototype only - not a medical device.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:              # import through the package path only, so each class exists once
    sys.path.insert(0, str(REPO))
from app.frontend import inputs as ui  # noqa: E402
from app.frontend.client import DEFAULT_URL, BackendClient, BackendError  # noqa: E402

WARNING = "Research prototype only — not a medical device. Do not use for diagnosis or treatment."
FIG = ui.REPO / "artifacts/reports/final_capstone/figures/fig_final_architecture_v2_render.png"
NAVY, GREEN, GRID = "#0F2A44", "#007F55", "#D5DDE5"

st.set_page_config(page_title="ECG selective completion · research demo", layout="wide")


def reset():
    for k in ("result", "result_sig", "error"):
        st.session_state.pop(k, None)
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1     # new keys: clears the upload and lead selection
    st.session_state["source"] = next(iter(ui.FIXTURES))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=8)


def plot_observed(df, observed):
    leads = [l for l in ui.LEAD_NAMES if l in observed]
    fig, axes = plt.subplots(len(leads), 1, figsize=(11, 0.95 * len(leads) + 0.5), sharex=True, squeeze=False)
    t = np.arange(len(df)) / ui.FS
    for ax, lead in zip(axes[:, 0], leads):
        ax.plot(t, df[lead].to_numpy(dtype=float), color=NAVY, linewidth=0.8)
        ax.set_ylabel(f"{lead} (mV)", rotation=0, ha="right", va="center", fontsize=9)
        _style(ax)
    axes[-1, 0].set_xlabel("time (s), 500 Hz", fontsize=9)
    fig.tight_layout()
    return fig


def plot_reconstruction(rec, observed):
    vals = np.asarray(rec["values"], dtype=float)
    t = np.arange(vals.shape[1]) / rec["sampling_rate_hz"]
    fig, axes = plt.subplots(12, 1, figsize=(11, 9.5), sharex=True)
    for ax, lead, v in zip(axes, rec["lead_names"], vals):
        obs = lead in observed
        ax.plot(t, v, color=NAVY if obs else GREEN, linewidth=0.8)
        ax.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=9, color=NAVY if obs else GREEN)
        ax.set_ylim(-0.05, 1.05)
        _style(ax)
    axes[-1].set_xlabel("time (s), 250 Hz; each lead min–max scaled to [0, 1] (the model's scale, not mV)", fontsize=9)
    axes[0].set_title("navy = observed lead, copied through exactly · green = reconstructed lead", fontsize=9, loc="left")
    fig.tight_layout()
    return fig


st.session_state.setdefault("nonce", 0)
client = st.session_state.get("backend_client")          # tests inject a stand-in here
if client is None:
    client = BackendClient(st.session_state.get("backend_url", DEFAULT_URL))

info, backend_problem = None, None
with st.sidebar:
    st.header("Backend")
    st.text_input("Local backend URL", value=DEFAULT_URL, key="backend_url")
    try:
        client.health()
        info = client.info()
    except BackendError as exc:
        backend_problem = str(exc)
    if info:
        st.success("Connected · frozen assets verified" if info.get("assets_verified") else "Connected")
        m = info["models"]
        st.markdown(f"**Reconstructor:** G v0.1 (`{m['reconstructor_sha256'][:12]}…`)  \n"
                    f"**Gate:** Gate B (`{m['gate_sha256'][:12]}…`)  \n"
                    f"**Calibrator:** Platt, 9a (`{m['calibrator_sha256'][:12]}…`)  \n"
                    f"**Threshold t:** {m['threshold']} (locked on 9b)")
        st.caption(m["not_used"])
    else:
        st.error(f"Backend not available: {backend_problem}")
    st.button("Reset", on_click=reset, key="reset", width="stretch")

st.title("Risk-Controlled Selective ECG Completion")
st.caption("BTech CSE (AI&ML) capstone · research demonstration of the frozen pipeline: G1/G2 features → Gate B → Platt "
           "calibration → locked threshold → ACCEPT or ABSTAIN; L2G G v0.1 reconstructs only after ACCEPT.")
st.error(WARNING, icon="⚠️")

tab_run, tab_about = st.tabs(["Run the demo", "About the model"])

with tab_run:
    if not info:
        st.info("Start the backend first (see app/README_APP.md), then reload this page.")
    else:
        nonce = st.session_state["nonce"]
        source = st.radio("Input", [*ui.FIXTURES, ui.SYNTHETIC, ui.UPLOAD], key="source")
        df, src_name = None, None
        try:
            if source in ui.FIXTURES:
                df, _ = ui.load_fixture(source)
                src_name = ui.FIXTURES[source].name
                st.caption("Synthetic signal generated by examples/synthetic/make_synthetic_ecg.py: NOT an ECG recording and "
                           "not patient data. It exercises the software paths only. The file holds only its observed leads.")
            elif source == ui.SYNTHETIC:
                df = ui.synthetic_frame()
                st.warning("Synthetic signal for exercising the workflow only. It is not an ECG recording, and the gate's "
                           "decision on it means nothing.")
            else:
                up = st.file_uploader("CSV: one column per observed lead (I, II, III, AVR, AVL, AVF, V1–V6), 500 Hz, mV, "
                                      "at least 4,096 rows (5,000 = 10 s). Processed in memory; never saved.",
                                      type=["csv"], key=f"upload_{nonce}")
                if up is not None:
                    ui.check_upload_name(up.name)
                    df = ui.frame_from_text(ui.read_csv_text(up.getvalue()))
                    src_name = up.name
        except ui.InputProblem as exc:
            st.error(f"Input refused: {exc}")
            df = None

        if df is not None:
            available = [l for l in ui.LEAD_NAMES if l in df.columns]
            observed = st.multiselect("Observed leads (the whole-lead mask; every other lead is treated as missing)",
                                      available, default=available, key=f"observed_{source}_{nonce}")
            supported = ui.is_supported_set(observed, info["lead_sets"])
            if observed and not supported:
                st.warning("This lead set is not one of the 258 L2G lead sets (any single lead, or 2–7 of I, III, V1–V6); "
                           "the service will refuse it.")
            sig = (source, src_name, tuple(observed), nonce)
            if st.session_state.get("result_sig") not in (None, sig):
                for k in ("result", "result_sig", "error"):
                    st.session_state.pop(k, None)
            if observed:
                st.subheader("Observed ECG (input)")
                fig = plot_observed(df, observed)
                st.pyplot(fig)
                plt.close(fig)
                missing = [l for l in ui.LEAD_NAMES if l not in observed]
                st.caption(f"{len(df):,} samples ({len(df) / ui.FS:.2f} s) · observed: {', '.join(observed)} · "
                           f"missing: {', '.join(missing) or 'none'}")
            if st.button("Run the gate", type="primary", key="run", disabled=not observed):
                try:
                    st.session_state["result"] = client.infer(ui.build_payload(df, observed, src_name))
                    st.session_state.pop("error", None)
                except (BackendError, ui.InputProblem) as exc:
                    st.session_state["error"] = (getattr(exc, "code", "INVALID_INPUT"), str(exc))
                    st.session_state.pop("result", None)
                st.session_state["result_sig"] = sig

        st.session_state["shown_reconstruction"] = False
        if st.session_state.get("error"):
            code, msg = st.session_state["error"]
            st.error(f"Refused ({code}): {msg}. Nothing was released.")
        res = st.session_state.get("result")
        if res:
            outcome, gate = res["outcome"], res["gate"]
            if outcome == "RELEASED":
                st.success("ACCEPT · the reconstruction is released")
            elif outcome == "ABSTAINED":
                st.warning("ABSTAIN · no reconstruction was computed")
            else:
                st.error("REFUSED · the reconstruction failed the release check; nothing is released")
            st.write(res["explanation"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Calibrated risk score", f"{gate['calibrated_score']:.4f}")
            c2.metric("Locked threshold t", f"{float(gate['threshold']):.4f}")
            c3.metric("Gate decision", gate["decision"])
            for w in res.get("warnings", []):
                st.info(f"Input note: {w}")
            if outcome == "RELEASED" and res.get("reconstruction"):
                st.subheader("Reconstruction (released)")
                fig = plot_reconstruction(res["reconstruction"], res["observed_leads"])
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["shown_reconstruction"] = True
            else:
                st.info("No reconstruction is shown for this input.")
            st.caption("ACCEPT is not evidence of diagnostic fidelity: Gate B withholds most bundle-branch-block ECGs, "
                       "and D1 is disagreement with one frozen classifier, not clinical harm.")
            with st.expander("Technical details"):
                details = {k: v for k, v in res.items() if k != "reconstruction"}
                if res.get("reconstruction"):
                    details["reconstruction"] = {k: v for k, v in res["reconstruction"].items() if k != "values"}
                st.json(details)

with tab_about:
    st.subheader("Audited architecture")
    if FIG.is_file():
        st.image(str(FIG), caption="Figure 1 (report v1.4): inference decides with the gate first; L2G runs only after "
                                   "ACCEPT. The research and evaluation paths are offline.", width="stretch")
    st.markdown("- **Reconstructor of record:** G v0.1 (L2G-ECG based). L2G v0.2-exp is exploratory and not used here.\n"
                "- **Gate inputs:** G1 mask geometry (15) and G2 observed-lead quality (15) only. The complete ECG, labels, "
                "D1 and E1 outputs are never inputs.\n"
                "- **Evaluation:** one pre-registered run on PTB-XL fold 10 (now spent); no external or clinical "
                "validation.\n"
                "- **Known limitation:** Gate B releases only 7.9 % of E1-called CLBBB rows; it is a deferral rule, not a "
                "safety claim.")
