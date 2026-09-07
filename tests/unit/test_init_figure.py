"""CPU regressions for the Smart Tool initial figure adapter."""

import numpy as np
import pytest
import supervisely as sly

from src.init_figure import (
    InitFigureError,
    bitmap_to_mask,
    download_init_mask,
    label_to_bitmap,
    label_to_mask,
)
from tests.unit.labels import bitmap_label, multipolygon_label, polygon_label, rect_ring


class FakeAnnotationApi:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def download_json(self, image_id):
        self.calls.append(image_id)
        return {"size": {"height": 20, "width": 30}, "objects": self.objects}


class FakeApi:
    def __init__(self, objects):
        self.annotation = FakeAnnotationApi(objects)


# --------------------------------------------------------------------------- #
# bitmap
# --------------------------------------------------------------------------- #
def test_bitmap_label_keeps_offset_and_data():
    data = np.ones((3, 4), dtype=bool)
    mask = label_to_mask(bitmap_label(data, row=5, col=7), 20, 30)

    assert mask.shape == (20, 30)
    assert mask[5:8, 7:11].all()
    assert mask.sum() == 12

    bitmap = label_to_bitmap(bitmap_label(data, row=5, col=7), 20, 30)
    assert (bitmap.origin.row, bitmap.origin.col) == (5, 7)
    assert bitmap.data.shape == (3, 4)


def test_bitmap_label_with_holes_is_preserved():
    data = np.ones((5, 5), dtype=bool)
    data[1:4, 1:4] = False
    data[2, 2] = True  # island inside the hole

    mask = label_to_mask(bitmap_label(data, row=2, col=3), 20, 30)

    assert mask[2:7, 3:8].sum() == data.sum()
    assert not mask[3, 4]
    assert mask[4, 5]


def test_bitmap_label_is_clipped_at_image_bounds():
    data = np.ones((6, 6), dtype=bool)

    # sticks out of the bottom-right corner
    mask = label_to_mask(bitmap_label(data, row=17, col=27), 20, 30)
    assert mask.sum() == 3 * 3
    assert mask[17:20, 27:30].all()

    # sticks out of the top-left corner
    mask = label_to_mask(bitmap_label(data, row=-2, col=-4), 20, 30)
    assert mask.sum() == 4 * 2
    assert mask[0:4, 0:2].all()


def test_legacy_bitmap_payload_without_geometry_type():
    label = bitmap_label(np.ones((2, 2), dtype=bool), row=1, col=1)
    label.pop("geometryType")
    label.pop("shape", None)

    mask = label_to_mask(label, 20, 30)
    assert mask[1:3, 1:3].all()
    assert mask.sum() == 4


def test_bitmap_label_without_data_raises():
    label = {"id": 1, "geometryType": "bitmap"}
    with pytest.raises(InitFigureError, match="no 'bitmap' field"):
        label_to_mask(label, 20, 30)

    label = {"id": 1, "geometryType": "bitmap", "bitmap": {"data": "not-a-bitmap", "origin": [0, 0]}}
    with pytest.raises(InitFigureError, match="Invalid 'bitmap' geometry"):
        label_to_mask(label, 20, 30)


# --------------------------------------------------------------------------- #
# polygon
# --------------------------------------------------------------------------- #
def test_polygon_with_hole_is_rasterized():
    label = polygon_label(
        exterior=rect_ring(2, 2, 12, 12),
        interior=[rect_ring(5, 5, 9, 9)],
    )

    mask = label_to_mask(label, 20, 30)

    assert mask.shape == (20, 30)
    assert mask[2, 2] and mask[12, 12] and mask[7, 3]
    assert not mask[7, 7]  # hole
    assert not mask[1, 1] and not mask[13, 13]
    # 11x11 filled square (boundary included) without the 5x5 hole
    assert mask.sum() == 11 * 11 - 5 * 5


def test_polygon_without_interior_field_is_accepted():
    label = polygon_label(exterior=rect_ring(1, 1, 4, 4))
    label["points"].pop("interior", None)

    mask = label_to_mask(label, 20, 30)
    assert mask.sum() == 4 * 4
    assert mask[1:5, 1:5].all()


def test_polygon_is_clipped_at_image_bounds():
    label = polygon_label(exterior=rect_ring(-5, -5, 3, 3))

    mask = label_to_mask(label, 20, 30)

    assert mask[0:4, 0:4].all()
    assert mask.sum() == 4 * 4


def test_polygon_float_coordinates_are_accepted():
    label = polygon_label(exterior=[[2.4, 2.7], [6.2, 2.1], [6.9, 6.3], [2.2, 6.8]])

    mask = label_to_mask(label, 20, 30)
    assert mask.any()
    assert mask[3, 3]


def test_polygon_outside_the_image_raises():
    label = polygon_label(exterior=rect_ring(100, 100, 120, 120))

    assert not label_to_mask(label, 20, 30).any()
    with pytest.raises(InitFigureError, match="empty after normalization"):
        label_to_bitmap(label, 20, 30)


@pytest.mark.parametrize(
    "label, message",
    [
        ({"id": 1, "geometryType": "polygon"}, "no 'points' field"),
        (polygon_label(exterior=[[1, 1], [2, 2]]), "at least 3 points"),
        (polygon_label(exterior="nope"), "at least 3 points"),
        (polygon_label(exterior=[[1, 1], [2, 2], [3]]), "pair of numbers"),
        (polygon_label(exterior=rect_ring(1, 1, 5, 5), interior="nope"), "must be a list of rings"),
        (
            polygon_label(exterior=rect_ring(1, 1, 5, 5), interior=[[[2, 2], [3, 3]]]),
            "at least 3 points",
        ),
    ],
)
def test_malformed_polygon_raises(label, message):
    with pytest.raises(InitFigureError, match=message):
        label_to_mask(label, 20, 30)


# --------------------------------------------------------------------------- #
# multipolygon
# --------------------------------------------------------------------------- #
def test_multipolygon_parts_are_unioned_and_holes_stay_local():
    label = multipolygon_label(
        parts=[
            {"exterior": rect_ring(0, 0, 10, 10), "interior": [rect_ring(3, 3, 7, 7)]},
            {"exterior": rect_ring(6, 6, 14, 14), "interior": []},
            {"exterior": rect_ring(20, 15, 25, 18), "interior": []},
        ]
    )

    mask = label_to_mask(label, 20, 30)

    assert mask[1, 1]  # first part
    assert not mask[4, 4]  # hole of the first part, not covered by any other part
    assert mask[6, 6]  # hole of the first part, but covered by the second part
    assert mask[12, 12]  # second part only
    assert mask[16, 21]  # disconnected third part
    assert not mask[15, 15]  # gap between the parts


def test_multipolygon_parts_may_be_nested_under_points():
    parts = [{"exterior": rect_ring(1, 1, 3, 3), "interior": []}]
    label = {"id": 7, "geometryType": "multipolygon", "points": {"parts": parts}}

    mask = label_to_mask(label, 20, 30)
    assert mask[1:4, 1:4].all()
    assert mask.sum() == 3 * 3


def test_multipolygon_is_clipped_at_image_bounds():
    label = multipolygon_label(
        parts=[
            {"exterior": rect_ring(25, 15, 40, 30), "interior": []},
            {"exterior": rect_ring(-4, -4, 2, 2), "interior": []},
        ]
    )

    mask = label_to_mask(label, 20, 30)

    assert mask[15:20, 25:30].all()
    assert mask[0:3, 0:3].all()
    assert mask.sum() == 5 * 5 + 3 * 3


@pytest.mark.parametrize(
    "label, message",
    [
        ({"id": 1, "geometryType": "multipolygon"}, "no 'parts' field"),
        (multipolygon_label(parts=[]), "no 'parts' field"),
        (multipolygon_label(parts=[[1, 2]]), r"'parts\[0\]' must be a dict"),
        (multipolygon_label(parts=[{"interior": []}]), "at least 3 points"),
    ],
)
def test_malformed_multipolygon_raises(label, message):
    with pytest.raises(InitFigureError, match=message):
        label_to_mask(label, 20, 30)


# --------------------------------------------------------------------------- #
# dispatch, AnyShape and unsupported geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label",
    [
        bitmap_label(np.ones((3, 3), dtype=bool), row=4, col=4),
        polygon_label(rect_ring(4, 4, 6, 6), class_title="any_shape"),
        multipolygon_label([{"exterior": rect_ring(4, 4, 6, 6), "interior": []}]),
    ],
)
def test_anyshape_concrete_geometries_are_dispatched_by_geometry_type(label):
    """AnyShape is a class: the concrete ``geometryType`` drives the conversion."""
    label = {**label, "classTitle": "any_shape_class"}

    bitmap = label_to_bitmap(label, 20, 30)

    assert isinstance(bitmap, sly.Bitmap)
    assert (bitmap.origin.row, bitmap.origin.col) == (4, 4)
    assert bitmap.data.shape == (3, 3)
    assert bitmap.data.all()


@pytest.mark.parametrize("geometry_type", ["rectangle", "point", "line", "alpha_mask", "graph"])
def test_unsupported_geometry_raises(geometry_type):
    label = {
        "id": 1,
        "geometryType": geometry_type,
        "points": {"exterior": rect_ring(1, 1, 5, 5), "interior": []},
    }

    with pytest.raises(InitFigureError, match="is not supported"):
        label_to_mask(label, 20, 30)


def test_label_without_geometry_type_raises():
    with pytest.raises(InitFigureError, match="no 'geometryType' field"):
        label_to_mask({"id": 1, "points": {"exterior": rect_ring(1, 1, 5, 5)}}, 20, 30)


def test_invalid_image_size_raises():
    with pytest.raises(InitFigureError, match="Invalid image size"):
        label_to_mask(polygon_label(rect_ring(1, 1, 5, 5)), 0, 30)


# --------------------------------------------------------------------------- #
# download_init_mask / bitmap_to_mask
# --------------------------------------------------------------------------- #
def test_download_init_mask_returns_bitmap_in_image_coordinates():
    api = FakeApi(
        objects=[
            bitmap_label(np.ones((2, 2), dtype=bool), row=0, col=0, figure_id=11),
            polygon_label(rect_ring(4, 3, 8, 9), figure_id=22),
        ]
    )

    bitmap = download_init_mask(api, 22, 101, 20, 30)

    assert api.annotation.calls == [101]
    assert isinstance(bitmap, sly.Bitmap)
    assert (bitmap.origin.row, bitmap.origin.col) == (3, 4)
    assert bitmap.data.shape == (7, 5)
    assert bitmap.data.all()


def test_download_init_mask_missing_figure_raises():
    api = FakeApi(objects=[polygon_label(rect_ring(1, 1, 5, 5), figure_id=22)])

    with pytest.raises(InitFigureError, match="not found in image"):
        download_init_mask(api, 33, 101, 20, 30)


def test_download_init_mask_without_figure_id_raises():
    api = FakeApi(objects=[])

    with pytest.raises(InitFigureError, match="'figure_id' is not defined"):
        download_init_mask(api, None, 101, 20, 30)
    assert api.annotation.calls == []


def test_bitmap_to_mask_places_bitmap_on_full_image():
    data = np.ones((3, 2), dtype=bool)
    bitmap = sly.Bitmap(data, origin=sly.PointLocation(row=6, col=9))

    mask = bitmap_to_mask(bitmap, 20, 30)

    assert mask.shape == (20, 30)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    assert (mask[6:9, 9:11] == 255).all()
    assert mask.sum() == 255 * 6


def test_bitmap_to_mask_clips_bitmaps_that_stick_out():
    data = np.ones((5, 5), dtype=bool)
    bitmap = sly.Bitmap(data, origin=sly.PointLocation(row=18, col=28))

    mask = bitmap_to_mask(bitmap, 20, 30)

    assert mask.shape == (20, 30)
    assert (mask[18:20, 28:30] == 255).all()
    assert mask.sum() == 255 * 4
