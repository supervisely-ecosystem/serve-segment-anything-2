"""CPU compatibility adapter for Smart Tool initial figures.

The SDK bundled into this app image (``supervisely==6.73.486``, see
``docker/Dockerfile``) decodes the initial figure of a Smart Tool request with
``sly.Bitmap.from_json`` only, so polygon and multipolygon figures (including
the concrete geometries of ``AnyShape`` classes) cannot be used as an initial
mask.  This module implements the shared normalization contract locally, with
the geometry primitives that are already available in that image, so no SDK
upgrade or image rebuild is required.

The adapter is intentionally free of any model/torch imports: it only converts
downloaded label JSON into a full-image CPU mask.
"""

from numbers import Real
from typing import Any, Dict, List, Optional

import numpy as np
import supervisely as sly

BITMAP = "bitmap"
POLYGON = "polygon"
MULTIPOLYGON = "multipolygon"

#: Concrete geometries accepted as a Smart Tool initial figure.
SUPPORTED_GEOMETRY_TYPES = (BITMAP, POLYGON, MULTIPOLYGON)


class InitFigureError(ValueError):
    """Raised when an initial figure cannot be converted into a mask.

    Missing figures, unsupported geometries and malformed geometry JSON are
    reported explicitly instead of being silently ignored or parsed as bitmaps.
    """


def _get_geometry_type(label: Dict[str, Any]) -> str:
    """Returns the concrete geometry type of a downloaded label.

    ``AnyShape`` is a class, not a geometry: labels of such classes are stored
    with their concrete ``geometryType``, so dispatching on the label is enough.
    """
    if not isinstance(label, dict):
        raise InitFigureError(f"Label must be a dict, got {type(label).__name__}.")
    geometry_type = label.get("geometryType")
    if geometry_type is None:
        # Legacy payloads may come without an explicit geometry type.
        if isinstance(label.get(BITMAP), dict):
            return BITMAP
        raise InitFigureError("Label has no 'geometryType' field.")
    if not isinstance(geometry_type, str):
        raise InitFigureError(f"Unsupported geometry type: {geometry_type}.")
    return geometry_type.lower()


def _place_bitmap(mask: np.ndarray, bitmap: sly.Bitmap) -> None:
    """Unions bitmap data into a full-image mask, clipping at image bounds."""
    data = bitmap.data
    if data.ndim != 2:
        raise InitFigureError("Bitmap data must be a 2-dimensional mask.")
    image_height, image_width = mask.shape[:2]
    top, left = bitmap.origin.row, bitmap.origin.col
    src_top, src_left = max(0, -top), max(0, -left)
    dst_top, dst_left = max(0, top), max(0, left)
    height = min(data.shape[0] - src_top, image_height - dst_top)
    width = min(data.shape[1] - src_left, image_width - dst_left)
    if height <= 0 or width <= 0:
        return
    region = data[src_top : src_top + height, src_left : src_left + width].astype(bool)
    mask[dst_top : dst_top + height, dst_left : dst_left + width] |= region


def _ring_to_points(ring: Any, name: str) -> List[sly.PointLocation]:
    if not isinstance(ring, (list, tuple)) or len(ring) < 3:
        raise InitFigureError(f"'{name}' must be a list of at least 3 points.")
    points = []
    for index, point in enumerate(ring):
        if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(
            isinstance(coord, Real) and not isinstance(coord, bool) for coord in point
        ):
            raise InitFigureError(f"'{name}' point at index {index} must be a pair of numbers.")
        # Supervisely stores vector geometry points as [x, y] pairs.
        x, y = point
        points.append(sly.PointLocation(row=y, col=x))
    return points


def _make_polygon(exterior: Any, interior: Any, name: str) -> sly.Polygon:
    if interior is None:
        interior = []
    if not isinstance(interior, (list, tuple)):
        raise InitFigureError(f"'{name}.interior' must be a list of rings.")
    try:
        return sly.Polygon(
            exterior=_ring_to_points(exterior, f"{name}.exterior"),
            interior=[
                _ring_to_points(ring, f"{name}.interior[{index}]")
                for index, ring in enumerate(interior)
            ],
        )
    except InitFigureError:
        raise
    except Exception as exc:  # malformed geometry, e.g. degenerate contours
        raise InitFigureError(f"Invalid '{name}' geometry: {exc}") from exc


def _polygon_parts(label: Dict[str, Any], geometry_type: str) -> List[sly.Polygon]:
    """Parses polygon / multipolygon label JSON into a list of polygon parts."""
    if geometry_type == POLYGON:
        points = label.get("points")
        if not isinstance(points, dict):
            raise InitFigureError("Polygon label has no 'points' field.")
        return [_make_polygon(points.get("exterior"), points.get("interior"), "points")]

    # Multipolygon parts are stored next to the geometry type, older/nested
    # payloads keep them inside "points".
    parts = label.get("parts")
    if parts is None and isinstance(label.get("points"), dict):
        parts = label["points"].get("parts")
    if not isinstance(parts, list) or len(parts) == 0:
        raise InitFigureError("Multipolygon label has no 'parts' field.")
    polygons = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise InitFigureError(f"'parts[{index}]' must be a dict.")
        polygons.append(
            _make_polygon(part.get("exterior"), part.get("interior"), f"parts[{index}]")
        )
    return polygons


def label_to_mask(label: Dict[str, Any], image_height: int, image_width: int) -> np.ndarray:
    """Converts a downloaded label into a full-image boolean mask.

    :param label: label JSON as returned by ``api.annotation.download_json``
    :param image_height: height of the image the mask is placed on
    :param image_width: width of the image the mask is placed on
    :return: boolean mask with the shape ``(image_height, image_width)``
    :raises InitFigureError: on unsupported or malformed geometry
    """
    if image_height <= 0 or image_width <= 0:
        raise InitFigureError(f"Invalid image size: {image_height}x{image_width}.")
    geometry_type = _get_geometry_type(label)
    if geometry_type not in SUPPORTED_GEOMETRY_TYPES:
        raise InitFigureError(
            f"Geometry '{geometry_type}' is not supported as a Smart Tool initial figure. "
            f"Supported geometries: {', '.join(SUPPORTED_GEOMETRY_TYPES)}."
        )

    mask = np.zeros((image_height, image_width), dtype=bool)
    if geometry_type == BITMAP:
        if not isinstance(label.get(BITMAP), dict):
            raise InitFigureError("Bitmap label has no 'bitmap' field.")
        try:
            bitmap = sly.Bitmap.from_json(label)
        except Exception as exc:
            raise InitFigureError(f"Invalid 'bitmap' geometry: {exc}") from exc
        _place_bitmap(mask, bitmap)
    else:
        # Every part is rasterized with its own holes and then unioned, so a
        # hole of one part cannot erase another part.
        for polygon in _polygon_parts(label, geometry_type):
            polygon.draw(mask, True)
    return mask


def label_to_bitmap(label: Dict[str, Any], image_height: int, image_width: int) -> sly.Bitmap:
    """Converts a downloaded label into a ``sly.Bitmap`` in image coordinates."""
    mask = label_to_mask(label, image_height, image_width)
    if not mask.any():
        raise InitFigureError(
            "Initial figure is empty after normalization: it does not intersect the image."
        )
    return sly.Bitmap(mask, extra_validation=False)


def download_init_mask(
    api: sly.Api,
    figure_id: Optional[int],
    image_id: int,
    image_height: int,
    image_width: int,
) -> sly.Bitmap:
    """Downloads the initial figure of an image and normalizes it to a bitmap.

    Drop-in replacement for
    ``supervisely.nn.inference.interactive_segmentation.functional.download_init_mask``
    that also accepts polygon and multipolygon labels. The image size is
    required to rasterize vector geometry in image coordinates.
    """
    if figure_id is None:
        raise InitFigureError("Initial figure is requested, but 'figure_id' is not defined.")
    ann_json = api.annotation.download_json(image_id)
    objects = (ann_json or {}).get("objects") or []
    labels = [
        label
        for label in objects
        if isinstance(label, dict) and label.get("id") == figure_id
    ]
    if len(labels) == 0:
        raise InitFigureError(f"Label with id {figure_id} not found in image {image_id}.")
    return label_to_bitmap(labels[0], image_height, image_width)


def bitmap_to_mask(bitmap: sly.Bitmap, image_height: int, image_width: int) -> np.ndarray:
    """Places a bitmap on a full-image ``uint8`` mask with 0 / 255 values.

    Unlike the SDK helper of the same name, bitmaps that stick out of the image
    are clipped instead of raising a broadcasting error.
    """
    mask = np.zeros((image_height, image_width), dtype=bool)
    _place_bitmap(mask, bitmap)
    return (mask * 255).astype(np.uint8)
