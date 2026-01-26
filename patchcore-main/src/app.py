import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from io import BytesIO

from models.patch_core import PatchCore
from models.patch_core import visualize


# ---------------------------
# CONFIG
# ---------------------------


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  
# Navigate: src/ -> patchcore-main/ -> patchcore-main/ -> repo root
DEFAULT_WEIGHTS = PROJECT_ROOT / "patchcore-main" / "data" / "weights" / "wide_resnet50_size224_param_0.1_9_wood.pth"
# ---------------------------
# MODEL LOADING
# ---------------------------
@st.cache_resource
def load_model(weights_path: str, device: str | None = None):
    weights_path = Path(weights_path).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"❌ Weights not found: {weights_path}")

    net = PatchCore.load_weights(str(weights_path), device)
    return net


# ---------------------------
# IMAGE HELPERS
# ---------------------------
def np_from_pil(pil: Image.Image) -> np.ndarray:
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def pil_from_np(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------
# DETECTION FUNCTION
# ---------------------------
def detect_image_from_pil(
    pil_img: Image.Image,
    net: PatchCore,
    th: float = 0.5,
    save_dir: str | None = None
):
    # Preprocess
    x = net.get_transform()(pil_img)
    x = net.get_resize()(x)
    x = torch.unsqueeze(x, 0).to(net.device)

    # Predict
    anomaly_score, anomaly_map, pred = net.predict(x, th=th)

    # Heatmap
    im_org = np_from_pil(pil_img)
    im_heatmap = visualize.create_heatmap_image(
        anomaly_map, org_size=im_org.shape
    )
    im_overlay = visualize.add_image(im_heatmap, im_org, alpha=0.5)

    # Annotation
    label = "DEFECT" if int(pred) == 1 else "OK"
    score_text = f"{float(anomaly_score):.4f}"
    color = (0, 0, 255) if int(pred) == 1 else (0, 255, 0)

    cv2.rectangle(im_overlay, (10, 10), (360, 70), (0, 0, 0), -1)
    cv2.putText(
        im_overlay,
        f"{label} | score: {score_text}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
        cv2.LINE_AA
    )

    # Save output
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "heatmap.png"), im_heatmap)
        cv2.imwrite(str(save_dir / "overlay.png"), im_overlay)

    return float(anomaly_score), int(pred), im_heatmap, im_overlay


# ---------------------------
# STREAMLIT UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="PatchCore Defect Detection",
        layout="wide"
    )

    st.title("🔍 PatchCore – Live Defect Detection")

    # Sidebar
    st.sidebar.header("Settings")

    weights_path = st.sidebar.text_input(
        "Weights path",
        value=str(DEFAULT_WEIGHTS)
    )

    threshold = st.sidebar.slider(
        "Decision Threshold",
        0.0, 1.0, 0.5, 0.01
    )

    save_outputs = st.sidebar.checkbox("Save outputs", False)
    output_dir = st.sidebar.text_input(
        "Output directory",
        value=str(PROJECT_ROOT / "data" / "output" / "streamlit")
    )

    st.sidebar.write("Weights exists:", Path(weights_path).exists())

    # Load model
    try:
        net = load_model(weights_path)
        st.sidebar.success("✅ Model loaded successfully")
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

    # ---------------------------
    # UPLOAD TAB
    # ---------------------------
    with tab1:
        uploaded = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded:
            pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
            st.image(pil_img, caption="Input Image", use_column_width=True)

            if st.button("Detect"):
                with st.spinner("Running PatchCore..."):
                    score, pred, heat, overlay = detect_image_from_pil(
                        pil_img,
                        net,
                        threshold,
                        output_dir if save_outputs else None
                    )

                st.metric("Anomaly Score", f"{score:.6f}")

                if pred == 1:
                    st.error("❗ DEFECT DETECTED")
                else:
                    st.success("✅ OK (No defect)")

                st.image(
                    cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                    caption="Heatmap",
                    use_column_width=True
                )
                st.image(
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    caption="Overlay",
                    use_column_width=True
                )

    # ---------------------------
    # CAMERA TAB
    # ---------------------------
    with tab2:
        cam = st.camera_input("Take a photo")

        if cam:
            pil_img = Image.open(cam).convert("RGB")
            st.image(pil_img, caption="Camera Image", use_column_width=True)

            if st.button("Detect (Camera)"):
                with st.spinner("Running PatchCore..."):
                    score, pred, heat, overlay = detect_image_from_pil(
                        pil_img,
                        net,
                        threshold,
                        output_dir if save_outputs else None
                    )

                st.metric("Anomaly Score", f"{score:.6f}")

                if pred == 1:
                    st.error("❗ DEFECT DETECTED")
                else:
                    st.success("✅ OK (No defect)")

                st.image(
                    cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                    caption="Heatmap",
                    use_column_width=True
                )
                st.image(
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    caption="Overlay",
                    use_column_width=True
                )

    st.markdown("---")
    st.markdown(
        "**Note:** This app uses a PatchCore memory bank trained on the *wood* dataset."
    )


if __name__ == "__main__":
    main()
