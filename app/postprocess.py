"""Turn raw YOLO segmentation results into an overlay image and detections."""
import base64
import io
from typing import List

import cv2
import numpy as np
from PIL import Image

from app.schemas import BoundingBox, Detection


def _color_for(index: int) -> np.ndarray:
    """Deterministic, distinct RGB color per instance (golden-ratio hue spacing)."""
    hue = int((index * 0.618033988749895 % 1.0) * 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def _full_res_masks(result, h: int, w: int) -> np.ndarray:
    """Return boolean masks of shape (N, h, w) in original image space."""
    if result.masks is None:
        return np.zeros((0, h, w), dtype=bool)
    # masks.xy holds polygons already scaled to the original image size,
    # so rasterising them avoids any letterbox / padding bookkeeping.
    out = np.zeros((len(result.masks.xy), h, w), dtype=np.uint8)
    for i, poly in enumerate(result.masks.xy):
        if len(poly) >= 3:
            cv2.fillPoly(out[i], [np.round(poly).astype(np.int32)], 1)
    return out.astype(bool)


def build_overlay(rgb: np.ndarray, result, alpha: float = 0.5) -> np.ndarray:
    """Blend instance masks onto the RGB image and draw boxes and labels."""
    if result.boxes is None or len(result.boxes) == 0:
        return rgb.copy()

    h, w = rgb.shape[:2]
    masks = _full_res_masks(result, h, w)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()

    overlay = rgb.astype(np.float32)
    for i, mask in enumerate(masks):
        overlay[mask] = overlay[mask] * (1 - alpha) + _color_for(i) * alpha
    out = np.ascontiguousarray(overlay.clip(0, 255).astype(np.uint8))

    thickness = max(1, round(min(h, w) / 400))
    font_scale = max(0.4, min(h, w) / 1200)
    for i, ((x1, y1, x2, y2), cls_id, conf) in enumerate(zip(boxes, class_ids, confs)):
        color = tuple(int(c) for c in _color_for(i))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        label = "{} {:.2f}".format(result.names[int(cls_id)], conf)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(p1[1], th + 6)
        cv2.rectangle(out, (p1[0], ty - th - 6), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(out, label, (p1[0] + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def extract_detections(result, h: int, w: int) -> List[Detection]:
    """Convert YOLO boxes and masks into validated Detection models."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    masks = _full_res_masks(result, h, w)
    detections = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=result.names[cls_id],
                confidence=round(float(box.conf.item()), 4),
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                mask_area_px=int(masks[i].sum()) if i < len(masks) else 0,
            )
        )
    return detections


def encode_png_base64(rgb: np.ndarray) -> str:
    """Encode an RGB array as a base64 PNG string for JSON transport."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
