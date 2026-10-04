"""Input validation and decoding for uploaded images."""
import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas import DecodedImage, UploadedImage


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def validate_upload(filename: str, content_type: str, data: bytes) -> UploadedImage:
    """Validate upload metadata (content type, size) with Pydantic."""
    return UploadedImage(
        filename=filename or "upload",
        content_type=content_type or "",
        size_bytes=len(data),
    )


def decode_image(data: bytes) -> Tuple[np.ndarray, DecodedImage]:
    """Decode bytes into an RGB numpy array and validate its properties.

    Returns the image as an HxWx3 uint8 RGB array plus its validated metadata.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated or corrupt files
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        raise ImageValidationError("File content is not a recognised image format")
    except (OSError, SyntaxError) as exc:
        raise ImageValidationError("Image file is corrupt or truncated: {}".format(exc))

    meta = DecodedImage(format=img.format or "", width=img.width, height=img.height, mode=img.mode)

    img = ImageOps.exif_transpose(img)  # respect camera orientation
    img = img.convert("RGB")  # normalise alpha, palette and grayscale images
    return np.asarray(img, dtype=np.uint8), meta


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    """Ultralytics expects numpy inputs in BGR order (OpenCV convention)."""
    return np.ascontiguousarray(rgb[:, :, ::-1])
