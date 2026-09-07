"""Builders for downloaded label JSON, shared by the offline unit tests."""

import numpy as np
import supervisely as sly


def bitmap_label(data: np.ndarray, row: int, col: int, figure_id: int = 1) -> dict:
    """Builds a bitmap label the way ``api.annotation.download_json`` returns it."""
    bitmap = sly.Bitmap(data.astype(bool), origin=sly.PointLocation(row=row, col=col))
    return {"id": figure_id, "classTitle": "mask", **bitmap.to_json()}


def rect_ring(left: int, top: int, right: int, bottom: int) -> list:
    """Closed rectangular ring in Supervisely's ``[x, y]`` point order."""
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def polygon_label(exterior, interior=None, figure_id: int = 1, class_title: str = "poly") -> dict:
    points = {"exterior": exterior}
    if interior is not None:
        points["interior"] = interior
    return {
        "id": figure_id,
        "classTitle": class_title,
        "geometryType": "polygon",
        "points": points,
    }


def multipolygon_label(parts, figure_id: int = 1, class_title: str = "multi") -> dict:
    return {
        "id": figure_id,
        "classTitle": class_title,
        "geometryType": "multipolygon",
        "parts": parts,
    }
