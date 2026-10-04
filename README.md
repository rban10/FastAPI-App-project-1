# FastAPI-App-project-1

A FastAPI wrapper around the YOLO11n instance segmentation model, with a Streamlit frontend.
Upload an image in the browser and get back the segmentation overlay plus a table of detections.

## Project structure

```
app/
  main.py         FastAPI app: model loading, GET /health, POST /predict
  preprocess.py   Upload validation, image decoding, EXIF rotation, RGB/BGR conversion
  postprocess.py  Mask rasterisation, overlay drawing, detection extraction, PNG encoding
  schemas.py      Pydantic models for params, input checks and the response
streamlit_app.py  Streamlit UI that calls the API
requirements.txt
```

## Setup

```bash
conda activate myenv2
pip install -r requirements.txt
```

For GPU inference, install a CUDA build of PyTorch first, for example:

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

The `yolo11n-seg.pt` weights download automatically on first start.
Set the `MODEL_PATH` environment variable to use a different checkpoint.

## Run

Start the API from the project root in one terminal:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Start the UI in a second terminal:

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 for the UI, or http://localhost:8000/docs for the interactive API docs.
Set `API_URL` if the API runs somewhere other than `http://localhost:8000`.

## API

### `POST /predict`

Multipart form request.

| Field   | Type  | Default | Rule                                   |
|---------|-------|---------|----------------------------------------|
| `file`  | file  | required | JPEG, PNG, BMP or WEBP, at most 10 MB |
| `conf`  | float | 0.25    | 0 to 1, confidence threshold           |
| `iou`   | float | 0.7     | 0 to 1, NMS IoU threshold              |
| `imgsz` | int   | 640     | 32 to 1920, multiple of 32             |
| `alpha` | float | 0.5     | 0 to 1, mask opacity in the overlay    |

```bash
curl -X POST http://localhost:8000/predict -F "file=@bus.jpg;type=image/jpeg" -F conf=0.3
```

Response:

```json
{
  "filename": "bus.jpg",
  "width": 810,
  "height": 1080,
  "num_detections": 6,
  "detections": [
    {
      "class_id": 5,
      "class_name": "bus",
      "confidence": 0.8985,
      "box": {"x1": 21.3, "y1": 229.9, "x2": 801.1, "y2": 744.2},
      "mask_area_px": 261956
    }
  ],
  "inference_ms": 125.6,
  "overlay_png_base64": "iVBORw0KGgo...",
  "message": null
}
```

### Validation

| Problem                                   | Status |
|-------------------------------------------|--------|
| Content type is not an allowed image type | 422    |
| Empty file or file over 10 MB             | 422    |
| Parameter out of range                    | 422    |
| Bytes are not a decodable image           | 400    |
| Truncated or corrupt image                | 400    |
| Width or height outside 32 to 8192 px     | 422    |

### `GET /health`

Returns `{"status": "ok", "model_loaded": true, "model": "yolo11n-seg.pt"}`.
