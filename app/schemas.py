"""Pydantic models for request validation and response serialization."""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/bmp", "image/webp")
ALLOWED_FORMATS = ("JPEG", "PNG", "BMP", "WEBP")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 32
MAX_DIMENSION = 8192


class PredictParams(BaseModel):
    """Inference parameters sent alongside the image."""

    model_config = ConfigDict(extra="forbid")

    conf: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou: float = Field(0.7, ge=0.0, le=1.0, description="NMS IoU threshold")
    imgsz: int = Field(640, ge=32, le=1920, description="Inference size, multiple of 32")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Mask overlay opacity")

    @field_validator("imgsz")
    @classmethod
    def imgsz_multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return v


class UploadedImage(BaseModel):
    """Metadata of the raw upload, validated before decoding."""

    filename: str = Field(..., min_length=1)
    content_type: str
    size_bytes: int = Field(..., gt=0, le=MAX_FILE_SIZE_BYTES)

    @field_validator("content_type")
    @classmethod
    def check_content_type(cls, v: str) -> str:
        v = (v or "").lower()
        if v not in ALLOWED_CONTENT_TYPES:
            raise ValueError(
                "Unsupported content type '{}'. Allowed: {}".format(v, ", ".join(ALLOWED_CONTENT_TYPES))
            )
        return v


class DecodedImage(BaseModel):
    """Properties of the decoded image, validated after decoding."""

    format: str
    width: int = Field(..., ge=MIN_DIMENSION, le=MAX_DIMENSION)
    height: int = Field(..., ge=MIN_DIMENSION, le=MAX_DIMENSION)
    mode: str

    @field_validator("format")
    @classmethod
    def check_format(cls, v: str) -> str:
        v = (v or "").upper()
        if v not in ALLOWED_FORMATS:
            raise ValueError("Decoded image format '{}' is not supported".format(v))
        return v


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox
    mask_area_px: int = Field(..., ge=0)


class PredictResponse(BaseModel):
    filename: str
    width: int
    height: int
    num_detections: int
    detections: List[Detection]
    inference_ms: float
    overlay_png_base64: str
    message: Optional[str] = None
