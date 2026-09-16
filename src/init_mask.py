"""Initial mask handling for Smart Tool requests (shared direct-mask contract).

On an initial ``Edit via Smart Tool`` request the labeling tool rasterizes the
edited figure (bitmap, polygon or multipolygon, including the concrete geometry
of an ``AnyShape`` class) and sends it directly::

    {"init_figure": true, "mask": {"origin": [x, y], "data": "<base64 bitmap>"}}

``origin`` is the top-left corner of the tight mask in full image / video frame
coordinates and ``data`` uses the encoded bitmap representation of Supervisely
``Bitmap`` JSON. A request carrying a mask is served without any annotation
API access, so it also works for callers that have no usable ``figure_id``.

The maskless ``figure_id`` download of the SDK is kept as a *deprecated*
compatibility path for legacy callers; invalid mask input is reported as an
explicit error and never falls back to that download silently.

This module intentionally stays free of model/torch imports: it only converts
the request payload into the full-image CPU mask the predictor expects.
"""

from numbers import Integral, Real
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.sly_logger import logger

#: Context field carrying the contract mask.
MASK_KEY = "mask"


class InitMaskError(ValueError):
    """Raised when the initial mask input cannot be used.

    Malformed, empty and out-of-bounds masks, as well as a missing initial
    figure input, are reported explicitly instead of being ignored.
    """


def _parse_origin(value: Any) -> Tuple[int, int]:
    """Validates the ``[x, y]`` origin of the contract mask."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise InitMaskError(
            f"'mask.origin' must be an [x, y] pixel pair, got {value!r}."
        )
    coordinates = []
    for name, coordinate in zip("xy", value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise InitMaskError(f"'mask.origin' {name} must be a number, got {coordinate!r}.")
        if not isinstance(coordinate, Integral) and float(coordinate) != int(coordinate):
            raise InitMaskError(
                f"'mask.origin' {name} must be a whole pixel coordinate, got {coordinate!r}."
            )
        coordinates.append(int(coordinate))
    return coordinates[0], coordinates[1]


def _decode_data(value: Any) -> np.ndarray:
    """Decodes the encoded bitmap payload of the contract mask."""
    if not isinstance(value, str) or not value.strip():
        raise InitMaskError("'mask.data' must be a non-empty base64 encoded bitmap.")
    try:
        data = np.asarray(sly.Bitmap.base64_2_data(value))
    except Exception as exc:
        raise InitMaskError(f"'mask.data' is not a valid encoded bitmap: {exc}") from exc
    if data.ndim != 2 or data.size == 0:
        raise InitMaskError(
            f"'mask.data' must decode to a non-empty 2-dimensional bitmap, got shape {data.shape}."
        )
    return data.astype(bool)


def place_on_image(
    data: np.ndarray,
    origin: Sequence[int],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Places a tight mask on a full-image boolean mask, clipping at the bounds.

    :param data: tight 2-dimensional mask
    :param origin: ``(x, y)`` position of the mask's top-left corner
    :param image_height: height of the image / video frame
    :param image_width: width of the image / video frame
    :return: boolean mask with the shape ``(image_height, image_width)``
    """
    if image_height <= 0 or image_width <= 0:
        raise InitMaskError(f"Invalid image size: {image_height}x{image_width}.")
    full_mask = np.zeros((image_height, image_width), dtype=bool)
    left, top = origin
    src_top, src_left = max(0, -top), max(0, -left)
    dst_top, dst_left = max(0, top), max(0, left)
    height = min(data.shape[0] - src_top, image_height - dst_top)
    width = min(data.shape[1] - src_left, image_width - dst_left)
    if height > 0 and width > 0:
        full_mask[dst_top : dst_top + height, dst_left : dst_left + width] = data[
            src_top : src_top + height, src_left : src_left + width
        ].astype(bool)
    return full_mask


def decode_context_mask(
    mask_json: Any, image_height: int, image_width: int
) -> sly.Bitmap:
    """Validates the contract mask and reconstructs it in image coordinates.

    :param mask_json: ``mask`` field of the Smart Tool context
    :param image_height: height of the image / video frame the mask belongs to
    :param image_width: width of the image / video frame the mask belongs to
    :return: bitmap positioned in image coordinates, clipped to the image
    :raises InitMaskError: on malformed, empty or out-of-bounds input
    """
    if not isinstance(mask_json, dict):
        raise InitMaskError(
            "'mask' must be an object with 'origin' and 'data', got "
            f"{type(mask_json).__name__}."
        )
    origin = _parse_origin(mask_json.get("origin"))
    data = _decode_data(mask_json.get("data"))
    full_mask = place_on_image(data, origin, image_height, image_width)
    if not full_mask.any():
        raise InitMaskError(
            f"Initial mask at origin {list(origin)} with shape {data.shape} has no "
            f"pixels inside the {image_height}x{image_width} image."
        )
    return sly.Bitmap(full_mask, extra_validation=False)


def bitmap_to_mask(
    bitmap: sly.Bitmap, image_height: int, image_width: int
) -> np.ndarray:
    """Renders a bitmap as the full-image ``uint8`` 0 / 255 predictor input.

    Unlike the SDK helper of the same name, bitmaps that stick out of the image
    are clipped instead of raising a broadcasting error.
    """
    full_mask = place_on_image(
        bitmap.data,
        (bitmap.origin.col, bitmap.origin.row),
        image_height,
        image_width,
    )
    return (full_mask * 255).astype(np.uint8)


def download_init_mask(
    api: sly.Api, figure_id: Optional[int], image_id: int
) -> sly.Bitmap:
    """Deprecated compatibility path for callers that send no ``mask``.

    Downloads the initial figure of an image by its id through the SDK, exactly
    as before the direct-mask contract. New callers must send ``mask`` instead:
    this path needs annotation API access and a usable ``figure_id``.
    """
    if figure_id is None:
        raise InitMaskError(
            "Initial figure is requested without a 'mask' and without a usable 'figure_id'."
        )
    logger.warn(
        "Smart Tool request has no 'mask': falling back to the deprecated "
        "figure_id annotation download.",
        extra={"figure_id": figure_id, "image_id": image_id},
    )
    try:
        return functional.download_init_mask(api, figure_id, image_id)
    except Exception as exc:
        raise InitMaskError(
            f"Could not download initial figure {figure_id} of image {image_id}: {exc}"
        ) from exc


def get_context_value(context: Dict[str, Any], *names: str) -> Any:
    """Reads the first present context field, supporting legacy spellings."""
    for name in names:
        if context.get(name) is not None:
            return context[name]
    return None
