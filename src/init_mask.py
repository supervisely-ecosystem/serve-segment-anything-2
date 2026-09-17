"""The Smart Tool's init mask: the mask of the figure that is being refined.

The app needs that mask as ``settings["init_mask"]``. Historically the request
only named the figure -- ``figure_id`` plus ``init_figure`` -- and the app
resolved it by downloading the image annotation and parsing the label back as a
``sly.Bitmap``. That restricts refinement to bitmap figures: a polygon,
multipolygon or AnyShape label cannot be parsed as a bitmap at all.

The platform now sends the figure's mask inline instead::

    "mask": {"data": <base64 string>, "origin": {"x": <int>, "y": <int>}}

``data`` is the encoding ``sly.Bitmap`` already uses on the wire (base64 of a
zlib-compressed PNG whose non-zero pixels are foreground, what
``sly.Bitmap.data_2_base64`` writes) and ``origin`` is the top-left corner of
the mask in full-image pixel coordinates, column (``x``) and row (``y``).

When the field is present it is authoritative: the init mask is built from it
alone, nothing is resolved through the API, and a mask that cannot be decoded is
a bad request rather than a silent fall back to the deprecated download.
"""

from typing import Any, Dict, Optional

import numpy as np
import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional

MASK_FIELD = "mask"


class InitMaskError(ValueError):
    """The request carries a ``mask`` field that cannot be used as an init mask."""


def init_bitmap_from_request(context: Dict[str, Any]) -> Optional[sly.Bitmap]:
    """Decode ``context["mask"]`` into the geometry the figure was drawn with.

    Returns ``None`` when the request carries no inline mask, in which case the
    caller falls back to the deprecated ``figure_id`` path. Raises
    :class:`InitMaskError` when a mask is present but unusable.
    """
    mask = context.get(MASK_FIELD)
    if mask is None:
        return None
    if not isinstance(mask, dict):
        raise InitMaskError(f"'{MASK_FIELD}' must be an object with 'data' and 'origin'.")
    data = mask.get("data")
    if not isinstance(data, str) or data == "":
        raise InitMaskError(f"'{MASK_FIELD}.data' must be a non-empty base64 string.")
    row, col = _origin_row_col(mask.get("origin"))
    try:
        # The same decoding sly.Bitmap.from_json applies to a stored figure, so
        # the geometry is indistinguishable from a downloaded one.
        decoded = sly.Bitmap.base64_2_data(data)
        return sly.Bitmap(decoded, origin=sly.PointLocation(row=row, col=col))
    except Exception as exc:
        raise InitMaskError(
            f"'{MASK_FIELD}.data' is not a decodable bitmap mask: {exc}"
        ) from exc


def build_init_mask(bitmap: sly.Bitmap, height: int, width: int) -> np.ndarray:
    """Rasterize the figure's mask onto the full inference image.

    Produces exactly what the ``figure_id`` path produces for the equivalent
    bitmap figure: an ``(height, width)`` ``uint8`` array with 0 outside and 255
    inside the figure.
    """
    bbox = bitmap.to_bbox()
    if bbox.top < 0 or bbox.left < 0 or bbox.bottom >= height or bbox.right >= width:
        raise InitMaskError(
            f"'{MASK_FIELD}' does not fit the image: mask at rows "
            f"{bbox.top}-{bbox.bottom}, columns {bbox.left}-{bbox.right} "
            f"in a {height}x{width} image."
        )
    return functional.bitmap_to_mask(bitmap, height, width)


def _origin_row_col(origin: Any) -> tuple:
    if not isinstance(origin, dict):
        raise InitMaskError(
            f"'{MASK_FIELD}.origin' must be an object with integer 'x' and 'y'."
        )
    try:
        col, row = int(origin["x"]), int(origin["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InitMaskError(
            f"'{MASK_FIELD}.origin' must be an object with integer 'x' and 'y'."
        ) from exc
    if row < 0 or col < 0:
        raise InitMaskError(f"'{MASK_FIELD}.origin' must not be negative.")
    return row, col
