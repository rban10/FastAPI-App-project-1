"""FastAPI service exposing YOLO11n instance segmentation at POST /predict."""
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from ultralytics import YOLO

from app.postprocess import build_overlay, encode_png_base64, extract_detections
from app.preprocess import ImageValidationError, decode_image, to_model_input, validate_upload
from app.schemas import MAX_FILE_SIZE_BYTES, PredictParams, PredictResponse

MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n-seg.pt")
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = YOLO(MODEL_PATH)  # downloads the weights on first run
    # Warm-up pass so the first real request does not pay CUDA / fusing startup cost.
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    models["seg"] = model
    yield
    models.clear()


app = FastAPI(
    title="YOLO11n Instance Segmentation API",
    description="Upload an image and get instance segmentation masks plus an overlay.",
    version="1.0.0",
    lifespan=lifespan,
)


def _validation_detail(exc: ValidationError):
    return [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]


def get_params(
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    imgsz: int = Form(640),
    alpha: float = Form(0.5),
) -> PredictParams:
    """Collect form fields and validate them through the PredictParams model."""
    try:
        return PredictParams(conf=conf, iou=iou, imgsz=imgsz, alpha=alpha)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "seg" in models, "model": MODEL_PATH}


def _run_pipeline(rgb, params: PredictParams):
    """Blocking inference + postprocessing, run off the event loop."""
    start = time.perf_counter()
    result = models["seg"].predict(
        to_model_input(rgb), conf=params.conf, iou=params.iou, imgsz=params.imgsz, verbose=False
    )[0]
    inference_ms = (time.perf_counter() - start) * 1000
    h, w = rgb.shape[:2]
    detections = extract_detections(result, h, w)
    overlay = build_overlay(rgb, result, alpha=params.alpha)
    return detections, overlay, inference_ms


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), params: PredictParams = Depends(get_params)):
    # Read at most one byte past the limit so oversized uploads fail validation cheaply.
    data = await file.read(MAX_FILE_SIZE_BYTES + 1)

    try:
        upload = validate_upload(file.filename, file.content_type, data)
        rgb, meta = decode_image(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc))
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if "seg" not in models:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    detections, overlay, inference_ms = await run_in_threadpool(_run_pipeline, rgb, params)

    return PredictResponse(
        filename=upload.filename,
        width=meta.width,
        height=meta.height,
        num_detections=len(detections),
        detections=detections,
        inference_ms=round(inference_ms, 1),
        overlay_png_base64=encode_png_base64(overlay),
        message=None if detections else "No objects detected",
    )
