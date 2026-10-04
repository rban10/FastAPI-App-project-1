"""Streamlit frontend: upload an image, call the FastAPI /predict endpoint, show the overlay."""
import base64
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="YOLO11n Segmentation", layout="wide")
st.title("YOLO11n Instance Segmentation")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.0, 1.0, 0.7, 0.05)
    imgsz = st.select_slider("Inference size", options=[320, 416, 512, 640, 800, 960, 1280], value=640)
    alpha = st.slider("Mask opacity", 0.0, 1.0, 0.5, 0.05)
    try:
        health = requests.get(API_URL + "/health", timeout=3).json()
        if health.get("model_loaded"):
            st.success("API online, model loaded")
        else:
            st.warning("API online, model not loaded")
    except requests.RequestException:
        st.error("API unreachable at " + API_URL)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    col1, col2 = st.columns(2)
    col1.subheader("Input")
    col1.image(uploaded.getvalue(), width="stretch")

    if st.button("Run segmentation", type="primary"):
        with st.spinner("Running inference..."):
            try:
                resp = requests.post(
                    API_URL + "/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"conf": conf, "iou": iou, "imgsz": imgsz, "alpha": alpha},
                    timeout=120,
                )
            except requests.RequestException as exc:
                st.error("Request failed: {}".format(exc))
                st.stop()

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            st.error("API error {}: {}".format(resp.status_code, detail))
            st.stop()

        result = resp.json()
        overlay_bytes = base64.b64decode(result["overlay_png_base64"])
        col2.subheader("Overlay")
        col2.image(overlay_bytes, width="stretch")
        col2.download_button("Download overlay", overlay_bytes, file_name="overlay.png", mime="image/png")

        st.caption(
            "{} detections in {} ms on a {}x{} image".format(
                result["num_detections"], result["inference_ms"], result["width"], result["height"]
            )
        )
        if result["detections"]:
            st.dataframe(
                [
                    {
                        "class": d["class_name"],
                        "confidence": d["confidence"],
                        "mask area (px)": d["mask_area_px"],
                        "box (x1, y1, x2, y2)": "{x1}, {y1}, {x2}, {y2}".format(**d["box"]),
                    }
                    for d in result["detections"]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(result.get("message") or "No objects detected")
