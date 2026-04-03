"""
PatchCore Surface Defect Detection
Compatible with: sujalgulhane/patchcore-main (wide_resnet50 · wood dataset)

Repo layout expected:
  patchcore-main/
  ├── src/
  │   ├── app.py              ← this file
  │   ├── models/patch_core/
  │   └── data/
  │       ├── weights/        ← .pth files go here
  │       └── output/
  └── requirements.txt

Run from repo root:
  cd src
  streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from io import BytesIO
import time

from models.patch_core import PatchCore
from models.patch_core import visualize

# ─────────────────────────────────────────────
# CONFIG  — portable path resolution
# Works for:
#   Local dev:        <repo>/src/app.py
#   Streamlit Cloud:  /mount/src/<repo>/<repo>/src/app.py
# ─────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent   # directory containing app.py

def _find_weights_dir() -> Path:
    """Walk up from app.py until we find a data/weights folder."""
    candidate = SRC_DIR
    for _ in range(6):
        w = candidate / "data" / "weights"
        if w.is_dir():
            return w
        candidate = candidate.parent
    # Fallback: create it next to app.py so the uploader can save there
    fallback = SRC_DIR / "data" / "weights"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

WEIGHTS_DIR = _find_weights_dir()
OUTPUT_DIR  = WEIGHTS_DIR.parent / "output" / "streamlit"

WEIGHT_FILES = {
    "Wide-ResNet50  (best accuracy)": "wide_resnet50_size224_param_0.1_9_wood.pth",
    "ResNet50       (balanced)"     : "resnet50_size224_param_0.1_9_wood.pth",
    "ResNet18       (fastest)"      : "resnet18_size224_param_0.1_9_wood.pth",
}

BACKBONE_OPTIONS = {
    label: str(WEIGHTS_DIR / fname)
    for label, fname in WEIGHT_FILES.items()
}

# ─────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PatchCore · Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.stDeployButton             { display: none; }

.stApp { background: #0b0d0e; color: #e8e4dc; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111416 !important;
    border-right: 1px solid #1f2428;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280 !important;
    font-family: 'IBM Plex Mono', monospace;
}
[data-testid="stSidebar"] input[type="text"] {
    background: #1a1d20 !important; border: 1px solid #2a2f35 !important;
    border-radius: 4px !important; color: #e8e4dc !important;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px !important;
}
[data-testid="stSidebar"] input[type="text"]:focus {
    border-color: #f0b429 !important;
    box-shadow: 0 0 0 2px rgba(240,180,41,0.15) !important;
}

/* Slider */
.stSlider > div > div > div > div { background: #f0b429 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #111416;
    border-bottom: 1px solid #1f2428; border-radius: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7280;
    border: none; border-bottom: 2px solid transparent; border-radius: 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 12px 22px; transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important; color: #f0b429 !important;
    border-bottom: 2px solid #f0b429 !important;
}

/* Button */
.stButton > button {
    background: #f0b429; color: #0b0d0e; border: none; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 10px 28px; width: 100%; transition: all 0.15s;
}
.stButton > button:hover  { background: #e0a31f; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2f35; border-radius: 6px;
    padding: 16px; background: #111416; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #f0b429; }
[data-testid="stFileUploadDropzone"] label { color: #6b7280 !important; font-size: 13px; }

/* Metrics */
[data-testid="stMetric"] {
    background: #111416; border: 1px solid #1f2428;
    border-radius: 6px; padding: 12px 16px;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.1em; text-transform: uppercase; color: #6b7280 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 20px !important; color: #e8e4dc !important;
}

/* Alerts */
.stAlert {
    border-radius: 4px; font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; border: none; padding: 8px 14px;
}

/* Images */
[data-testid="stImage"] img { border-radius: 6px; border: 1px solid #1f2428; width: 100%; }

/* Column label */
.img-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: #4b5563; margin-bottom: 5px;
}

/* History badges */
.badge-defect {
    background: #1e0f0f; color: #f09595;
    border: 1px solid #e24b4a; border-radius: 3px; padding: 2px 8px; font-size: 11px;
}
.badge-pass {
    background: #0a1a10; color: #5dcaa5;
    border: 1px solid #1d9e75; border-radius: 3px; padding: 2px 8px; font-size: 11px;
}

.stSpinner > div { border-top-color: #f0b429 !important; }
[data-testid="stCheckbox"] label {
    font-size: 12px; color: #9ca3af; font-family: 'IBM Plex Mono', monospace;
}
[data-testid="stSelectbox"] > div > div {
    background: #1a1d20 !important; border: 1px solid #2a2f35 !important;
    color: #e8e4dc !important; font-family: 'IBM Plex Mono', monospace; font-size: 12px;
}
hr { border-color: #1f2428; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _label(text: str):
    st.markdown(f"<p class='img-label'>{text}</p>", unsafe_allow_html=True)


def _section(title: str):
    st.markdown(
        f"<p style='font-family:IBM Plex Mono,monospace;font-size:10px;"
        f"letter-spacing:.12em;text-transform:uppercase;color:#374151;"
        f"border-top:1px solid #1f2428;padding-top:14px;margin-bottom:10px'>"
        f"{title}</p>",
        unsafe_allow_html=True,
    )


def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def np_from_pil(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> PatchCore:
    p = Path(weights_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    return PatchCore.load_weights(str(p))


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(
    pil_img: Image.Image,
    net: PatchCore,
    th: float,
    alpha: float,
):
    x = net.get_transform()(pil_img)
    x = net.get_resize()(x)
    x = torch.unsqueeze(x, 0).to(net.device)

    t0 = time.perf_counter()
    anomaly_score, anomaly_map, pred = net.predict(x, th=th)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    im_org     = np_from_pil(pil_img)
    im_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)
    im_overlay = visualize.add_image(im_heatmap, im_org, alpha=alpha)

    # Burn-in verdict label on overlay
    label     = "DEFECT" if int(pred) == 1 else "OK"
    color_bgr = (0, 0, 220) if int(pred) == 1 else (0, 210, 80)
    cv2.rectangle(im_overlay, (8, 8), (370, 66), (11, 13, 14), -1)
    cv2.putText(
        im_overlay,
        f"{label}   score={float(anomaly_score):.4f}",
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2, cv2.LINE_AA,
    )

    return float(anomaly_score), int(pred), im_heatmap, im_overlay, elapsed_ms


# ─────────────────────────────────────────────
# SESSION STATE  – inspection history
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []


# ─────────────────────────────────────────────
# RESULT DISPLAY  (3-col grid, no scroll)
# ─────────────────────────────────────────────
def display_results(pil_img, net, threshold, alpha, save_outputs, output_dir, key_suffix=""):
    # Run button above the grid so it's always visible
    btn = st.button("▶  Run Detection", key=f"detect_{key_suffix}")

    # 3-column image grid
    c_in, c_heat, c_over = st.columns(3, gap="small")

    with c_in:
        _label("Input image")
        st.image(pil_img, use_container_width=True)

    with c_heat:
        _label("Anomaly heatmap")
        heat_slot = st.empty()

    with c_over:
        _label("Defect overlay")
        over_slot = st.empty()

    # Metrics row
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns([1.3, 1, 1, 1.7])
    score_slot   = m1.empty()
    verdict_slot = m2.empty()
    time_slot    = m3.empty()
    status_slot  = m4.empty()

    if btn:
        with st.spinner("Running PatchCore inference…"):
            score, pred, heat, overlay, elapsed = run_inference(
                pil_img, net, threshold, alpha
            )

        heat_slot.image(bgr_to_rgb(heat),    use_container_width=True)
        over_slot.image(bgr_to_rgb(overlay), use_container_width=True)

        verdict = "DEFECT" if pred == 1 else "PASS"
        score_slot.metric("Anomaly Score", f"{score:.6f}")
        verdict_slot.metric("Verdict",     verdict)
        time_slot.metric("Inference",      f"{elapsed:.0f} ms")

        if pred == 1:
            status_slot.error("⬛  Defect detected — check overlay for anomaly region")
        else:
            status_slot.success("⬛  No defect — surface within normal tolerance")

        # Optional save
        if save_outputs:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            pil_img.save(str(out / f"input_{ts}.png"))
            cv2.imwrite(str(out / f"heatmap_{ts}.png"), heat)
            cv2.imwrite(str(out / f"overlay_{ts}.png"), overlay)
            st.toast(f"Saved → {output_dir}", icon="💾")

        # History entry
        st.session_state.history.append({
            "thumb"  : pil_img.copy().resize((64, 64)),
            "score"  : score,
            "verdict": verdict,
            "elapsed": elapsed,
        })


# ─────────────────────────────────────────────
# HISTORY TAB
# ─────────────────────────────────────────────
def render_history():
    h = st.session_state.history
    if not h:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:12px;"
            "color:#4b5563;padding:20px 0'>No inspections yet.</p>",
            unsafe_allow_html=True,
        )
        return

    total   = len(h)
    defects = sum(1 for r in h if r["verdict"] == "DEFECT")
    avg_ms  = sum(r["elapsed"] for r in h) / total

    s1, s2, s3 = st.columns(3)
    s1.metric("Total inspected", total)
    s2.metric("Defects found",   defects)
    s3.metric("Avg inference",   f"{avg_ms:.0f} ms")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Table header
    h1, h2, h3, h4 = st.columns([0.8, 1.5, 1, 1])
    for col, hdr in zip([h1, h2, h3, h4], ["Thumb", "Score", "Verdict", "ms"]):
        col.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:10px;"
            f"letter-spacing:.08em;text-transform:uppercase;color:#374151'>{hdr}</p>",
            unsafe_allow_html=True,
        )

    for row in reversed(h):
        c0, c1, c2, c3 = st.columns([0.8, 1.5, 1, 1])
        c0.image(row["thumb"], width=56)
        c1.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:13px;"
            f"color:#e8e4dc;padding-top:14px'>{row['score']:.6f}</p>",
            unsafe_allow_html=True,
        )
        badge = "badge-defect" if row["verdict"] == "DEFECT" else "badge-pass"
        c2.markdown(
            f"<p style='padding-top:14px'><span class='{badge}'>{row['verdict']}</span></p>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:13px;"
            f"color:#9ca3af;padding-top:14px'>{row['elapsed']:.0f}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Clear history", key="clear_hist"):
        st.session_state.history.clear()
        st.rerun()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 22px;">
            <p style="font-family:'IBM Plex Mono',monospace;font-size:11px;
               letter-spacing:.18em;text-transform:uppercase;color:#f0b429;margin:0 0 4px">
               PatchCore</p>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:10px;
               letter-spacing:.06em;color:#4b5563;margin:0">
               Surface defect detection · Wood dataset</p>
        </div>""", unsafe_allow_html=True)

        _section("Model")
        backbone_label = st.selectbox("Backbone", list(BACKBONE_OPTIONS.keys()), index=0)
        weights_path   = BACKBONE_OPTIONS[backbone_label]

        exists = Path(weights_path).exists()
        st.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            f"color:{'#1d9e75' if exists else '#e24b4a'};margin-top:4px'>"
            f"{'✓ weights found at ' + str(WEIGHTS_DIR) if exists else '✗ not found — upload below'}</p>",
            unsafe_allow_html=True,
        )

        if not exists:
            st.markdown(
                "<p style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                "letter-spacing:.06em;text-transform:uppercase;color:#6b7280;"
                "margin-top:10px;margin-bottom:4px'>Upload .pth weights file</p>",
                unsafe_allow_html=True,
            )
            uploaded_weights = st.file_uploader(
                "Upload weights", type=["pth"], label_visibility="collapsed"
            )
            if uploaded_weights is not None:
                save_path = WEIGHTS_DIR / uploaded_weights.name
                WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(uploaded_weights.read())
                st.markdown(
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
                    f"color:#1d9e75'>✓ saved · reload to use</p>",
                    unsafe_allow_html=True,
                )
                st.rerun()

        _section("Inference")
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                              help="Scores above this are classified as defects.")
        st.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            f"color:#9ca3af;margin-top:-4px'>threshold = {threshold:.2f}</p>",
            unsafe_allow_html=True,
        )
        alpha = st.slider("Heatmap blend alpha", 0.1, 0.9, 0.5, 0.05,
                          help="Heatmap opacity in the overlay image.")

        _section("Output")
        save_outputs = st.checkbox("Save result images", value=False)
        output_dir   = st.text_input(
            "Output directory",
            value=str(OUTPUT_DIR),
            disabled=not save_outputs,
        )

        _section("Device")
        device_label = (
            "CUDA · " + torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "CPU (no GPU detected)"
        )
        st.markdown(
            f"<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            f"color:#9ca3af'>{device_label}</p>",
            unsafe_allow_html=True,
        )

    return weights_path, backbone_label, threshold, alpha, save_outputs, output_dir


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    weights_path, backbone_label, threshold, alpha, save_outputs, output_dir = render_sidebar()

    # Header
    st.markdown("""
    <div style="padding:22px 0 16px;border-bottom:1px solid #1f2428;margin-bottom:22px;">
        <h1 style="font-family:'IBM Plex Mono',monospace;font-size:20px;
           font-weight:600;letter-spacing:.04em;color:#e8e4dc;margin:0 0 5px">
           Surface Inspection System
        </h1>
        <p style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
           color:#6b7280;margin:0;font-weight:300">
           PatchCore memory-bank anomaly detection · wide_resnet50 backbone · wood texture dataset
        </p>
    </div>""", unsafe_allow_html=True)

    # Load model (cached)
    with st.spinner("Loading model weights…"):
        try:
            net = load_model(weights_path)
        except FileNotFoundError:
            wf = WEIGHT_FILES[backbone_label]
            st.markdown(f"""
            <div style="background:#1e0f0f;border-left:3px solid #e24b4a;
                 border-radius:4px;padding:16px 20px;margin-bottom:24px">
                <p style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                   color:#f09595;margin:0 0 10px;font-weight:600">
                   ⚠ Weights file not found</p>
                <p style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                   color:#9ca3af;margin:0 0 6px">
                   Expected: <code style="color:#f0b429">{weights_path}</code></p>
                <p style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                   color:#9ca3af;margin:0 0 10px">
                   Download <strong style="color:#e8e4dc">{wf}</strong> from the
                   <a href="https://github.com/ComputermindCorp/assets/releases/tag/v1.0.0"
                      style="color:#f0b429">ComputermindCorp releases page</a>
                   and upload it using the sidebar uploader, or commit it to
                   <code style="color:#f0b429">src/data/weights/</code> in your repo.</p>
            </div>""", unsafe_allow_html=True)
            st.stop()
        except Exception as e:
            st.markdown(f"""
            <div style="background:#1e0f0f;border-left:3px solid #e24b4a;
                 border-radius:4px;padding:14px 18px;margin-bottom:24px">
                <p style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                   color:#f09595;margin:0">⚠ Model error: {e}</p>
            </div>""", unsafe_allow_html=True)
            st.stop()

    tab_upload, tab_camera, tab_history = st.tabs([
        "📁  Upload image",
        "📷  Camera",
        "📋  Inspection history",
    ])

    # Upload tab
    with tab_upload:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop an image or click to browse",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed",
        )
        if uploaded:
            pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            display_results(pil_img, net, threshold, alpha,
                            save_outputs, output_dir, key_suffix="upload")

    # Camera tab
    with tab_camera:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        enable_cam = st.checkbox("Enable camera", value=False)
        if not enable_cam:
            st.markdown(
                "<p style='font-family:IBM Plex Mono,monospace;font-size:12px;"
                "color:#4b5563;margin-top:8px'>"
                "Camera is off · check the box above to activate</p>",
                unsafe_allow_html=True,
            )
        else:
            cam = st.camera_input("Capture", label_visibility="collapsed")
            if cam:
                pil_img = Image.open(cam).convert("RGB")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                display_results(pil_img, net, threshold, alpha,
                                save_outputs, output_dir, key_suffix="camera")

    # History tab
    with tab_history:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        render_history()

    # Footer
    st.markdown("""
    <div style="border-top:1px solid #1f2428;margin-top:32px;padding-top:12px;
         display:flex;justify-content:space-between;">
        <p style="font-family:'IBM Plex Mono',monospace;font-size:10px;
           letter-spacing:.06em;color:#374151;margin:0">
           PatchCore · Roth et al. 2021 · arxiv:2106.08265
        </p>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:10px;
           letter-spacing:.06em;color:#374151;margin:0">
           wide_resnet50 · 224px · wood dataset
        </p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
