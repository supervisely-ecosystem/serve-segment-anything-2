"""Regressions for the production ``/smart_segmentation`` request flow.

The handler that is registered by ``src.main`` is executed here directly; only
the external boundaries (Supervisely API, image cache and the model's
``predict``) are replaced, so request parsing, initial figure normalization,
predictor hand-off and bitmap response transport are the real code.
"""

import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import supervisely as sly
from cacheout import Cache
from cachetools import LRUCache
from fastapi import Response

from src import init_figure
from src.smart_segmentation import smart_segmentation
from tests.unit.labels import bitmap_label, multipolygon_label, polygon_label, rect_ring

IMAGE_ID = 777
FIGURE_ID = 4242
IMAGE_HEIGHT, IMAGE_WIDTH = 40, 60


def test_handler_does_not_pull_in_the_model_stack():
    """The offline suite must not import/initialize the model stack."""
    assert "sam2" not in sys.modules
    assert "torch" not in sys.modules
    assert "src.main" not in sys.modules


class FakeImageApi:
    def __init__(self, image):
        self.image = image
        self.download_np_calls = []

    def download_np(self, image_id):
        self.download_np_calls.append(image_id)
        return self.image


class FakeAnnotationApi:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def download_json(self, image_id):
        self.calls.append(image_id)
        return {"objects": list(self.objects)}


class FakeApi:
    """Only the endpoints the Smart Tool flow is allowed to use exist here."""

    def __init__(self, image, objects):
        self.image = FakeImageApi(image)
        self.annotation = FakeAnnotationApi(objects)


class FakeImageCache:
    """Stands in for ``sly.nn.inference.Cache`` (network boundary)."""

    def __init__(self, image):
        self._image = image
        self.download_image_calls = []
        self.download_frame_calls = []

    def download_image(self, api, image_id, related=False):
        self.download_image_calls.append(image_id)
        return self._image

    def download_frame(self, api, video_id, frame_index):
        self.download_frame_calls.append((video_id, frame_index))
        return self._image

    def download_image_by_hash(self, api, image_hash):
        return self._image


class StubModel:
    """Model stand-in with the real caches/locks used by the handler."""

    def __init__(self, image, pred_mask, use_bbox=True):
        self.cache = FakeImageCache(image)
        self.process_volume = None
        self.use_bbox = SimpleNamespace(is_switched=lambda: use_bbox)
        self._inference_image_cache = Cache(ttl=60)
        self._init_mask_cache = LRUCache(maxsize=100)
        self._inference_image_lock = threading.Lock()
        self._pred_mask = pred_mask
        self.predict_calls = []

    def _get_inference_settings(self, state):
        return dict(state.get("settings", {}))

    def predict(self, image_path, settings):
        # the image must still be on disk when the predictor is called
        assert os.path.isfile(image_path), image_path
        self.predict_calls.append({**settings, "image_path": image_path})
        return [SimpleNamespace(mask=self._pred_mask)]


def make_image():
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    return image


def make_request(context, state=None, api=None):
    return SimpleNamespace(
        state=SimpleNamespace(context=context, state=state or {}, api=api)
    )


def smart_tool_context(**overrides):
    context = {
        "image_id": IMAGE_ID,
        "figure_id": FIGURE_ID,
        "positive": [{"x": 10, "y": 12}],
        "negative": [],
    }
    context.update(overrides)
    return context


def call_handler(model, api, context, state=None):
    response = Response()
    result = smart_segmentation(model, response, make_request(context, state, api))
    return response, result


def decode(result):
    return sly.Bitmap.base64_2_data(result["bitmap"])


@pytest.fixture
def image():
    return make_image()


@pytest.fixture
def pred_mask():
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask[10:15, 20:24] = 255
    return mask


# --------------------------------------------------------------------------- #
# initial figure -> predictor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label, expected_pixels, unexpected_pixels",
    [
        pytest.param(
            bitmap_label(np.ones((6, 8), dtype=bool), row=5, col=7, figure_id=FIGURE_ID),
            [(5, 7), (10, 14)],
            [(4, 7), (11, 15)],
            id="bitmap",
        ),
        pytest.param(
            polygon_label(
                exterior=rect_ring(7, 5, 14, 10),
                interior=[rect_ring(9, 7, 12, 8)],
                figure_id=FIGURE_ID,
            ),
            [(5, 7), (10, 14), (6, 8)],
            [(4, 7), (11, 15), (8, 10)],
            id="polygon-with-hole",
        ),
        pytest.param(
            multipolygon_label(
                parts=[
                    {"exterior": rect_ring(7, 5, 14, 10), "interior": [rect_ring(9, 7, 12, 8)]},
                    {"exterior": rect_ring(20, 20, 30, 30), "interior": []},
                ],
                figure_id=FIGURE_ID,
            ),
            [(5, 7), (10, 14), (25, 25)],
            [(4, 7), (8, 10), (18, 18)],
            id="multipolygon-with-hole-and-disconnected-part",
        ),
    ],
)
def test_init_figure_is_normalized_and_handed_to_the_predictor(
    app_data_dir, image, pred_mask, label, expected_pixels, unexpected_pixels
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [label])

    response, result = call_handler(
        model, api, smart_tool_context(init_figure=True), state={"settings": {}}
    )

    assert response.status_code == 200
    assert api.annotation.calls == [IMAGE_ID]
    assert len(model.predict_calls) == 1

    init_mask = model.predict_calls[0]["init_mask"]
    assert init_mask.dtype == np.uint8
    assert init_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert set(np.unique(init_mask)) <= {0, 255}
    for row, col in expected_pixels:
        assert init_mask[row, col] == 255, (row, col)
    for row, col in unexpected_pixels:
        assert init_mask[row, col] == 0, (row, col)

    # the normalized figure is cached as a bitmap in image coordinates
    cached = model._init_mask_cache[FIGURE_ID]
    assert isinstance(cached, sly.Bitmap)
    assert init_figure.bitmap_to_mask(cached, IMAGE_HEIGHT, IMAGE_WIDTH).tolist() == (
        init_mask.tolist()
    )

    # response transport is unchanged: origin + encoded bitmap of the prediction
    assert result["success"] is True and result["error"] is None
    assert result["origin"] == {"x": 20, "y": 10}
    assert decode(result).tolist() == (pred_mask[10:15, 20:24] > 0).tolist()


def test_predictor_receives_points_and_full_image_settings(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [polygon_label(rect_ring(7, 5, 14, 10), figure_id=FIGURE_ID)])

    context = smart_tool_context(
        init_figure=True,
        positive=[{"x": 10, "y": 12}, {"x": 11, "y": 13}],
        negative=[{"x": 30, "y": 31}],
    )
    _, result = call_handler(model, api, context, state={"settings": {"conf": 0.5}})

    settings = model.predict_calls[0]
    assert settings["conf"] == 0.5
    assert settings["mode"] == "points"
    assert settings["input_image_id"] == IMAGE_ID
    assert "bbox_coordinates" not in settings
    assert settings["point_coordinates"] == [[10, 12], [11, 13], [30, 31]]
    assert settings["point_labels"] == [1, 1, 0]
    assert result["success"] is True


def test_continuation_request_reuses_cached_init_mask(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(
        image,
        [polygon_label(rect_ring(7, 5, 14, 10), figure_id=FIGURE_ID)],
    )

    call_handler(model, api, smart_tool_context(init_figure=True), state={"settings": {}})
    first_init_mask = model.predict_calls[0]["init_mask"]

    # the label is removed from the annotation: a continuation click must not
    # download it again, it works on the cached initial mask
    api.annotation.objects = []
    _, result = call_handler(
        model,
        api,
        smart_tool_context(positive=[{"x": 10, "y": 12}, {"x": 15, "y": 16}]),
        state={"settings": {}},
    )

    assert api.annotation.calls == [IMAGE_ID]  # no second download
    assert model.cache.download_image_calls == [IMAGE_ID]  # image cache reused
    assert len(model.predict_calls) == 2
    assert model.predict_calls[1]["init_mask"].tolist() == first_init_mask.tolist()
    assert result["success"] is True


def test_request_without_figure_has_no_init_mask(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [])

    context = smart_tool_context()
    context.pop("figure_id")
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert api.annotation.calls == []
    assert model.predict_calls[0]["init_mask"] is None
    assert result["success"] is True


def test_video_context_does_not_use_init_figure(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [])

    context = {
        "video": {"video_id": 5, "frame_index": 3},
        "figure_id": FIGURE_ID,
        "init_figure": True,
        "positive": [{"x": 10, "y": 12}],
        "negative": [],
    }
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert api.annotation.calls == []
    assert model.cache.download_frame_calls == [(5, 3)]
    assert model.predict_calls[0]["init_mask"] is None
    assert model.predict_calls[0]["input_image_id"] == "5_3"
    assert result["success"] is True


# --------------------------------------------------------------------------- #
# explicit errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "objects, message",
    [
        pytest.param(
            [{"id": FIGURE_ID, "geometryType": "rectangle", "points": {"exterior": rect_ring(1, 1, 5, 5)}}],
            "is not supported",
            id="unsupported-geometry",
        ),
        pytest.param([], "not found in image", id="missing-figure"),
        pytest.param(
            [polygon_label(exterior=[[1, 1], [2, 2]], figure_id=FIGURE_ID)],
            "at least 3 points",
            id="malformed-geometry",
        ),
        pytest.param(
            [polygon_label(rect_ring(200, 200, 220, 220), figure_id=FIGURE_ID)],
            "empty after normalization",
            id="outside-the-image",
        ),
    ],
)
def test_unusable_init_figure_returns_an_explicit_error(
    app_data_dir, image, pred_mask, objects, message
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, objects)

    response, result = call_handler(
        model, api, smart_tool_context(init_figure=True), state={"settings": {}}
    )

    assert response.status_code == 400
    assert result["success"] is False
    assert message in result["error"]
    assert result["bitmap"] is None and result["origin"] is None
    assert model.predict_calls == []
    assert FIGURE_ID not in model._init_mask_cache
    # the temporary image of the failed request is cleaned up
    assert os.listdir(app_data_dir) == []


# --------------------------------------------------------------------------- #
# crop, clicks and response transport
# --------------------------------------------------------------------------- #
def test_crop_is_forwarded_as_bbox_and_response_origin_is_in_image_coordinates(
    app_data_dir, image
):
    pred_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    pred_mask[12:16, 22:25] = 255
    pred_mask[30, 50] = 255  # outside of the crop: must not be returned
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [polygon_label(rect_ring(7, 5, 14, 10), figure_id=FIGURE_ID)])

    context = smart_tool_context(
        init_figure=True,
        positive=[{"x": 22, "y": 12}],
        crop=[{"x": 20, "y": 10}, {"x": 29, "y": 19}],
    )
    _, result = call_handler(model, api, context, state={"settings": {}})

    settings = model.predict_calls[0]
    assert settings["mode"] == "combined"
    assert settings["bbox_coordinates"] == [10, 20, 20, 30]
    assert settings["bbox_class_name"] == "target"
    # clicks stay in image coordinates
    assert settings["point_coordinates"] == [[22, 12]]
    # the initial mask keeps full image dimensions even with a crop
    assert settings["init_mask"].shape == (IMAGE_HEIGHT, IMAGE_WIDTH)

    assert result["origin"] == {"x": 22, "y": 12}
    assert decode(result).tolist() == np.ones((4, 3), dtype=bool).tolist()


def test_points_mode_when_bbox_switch_is_off(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask, use_bbox=False)
    api = FakeApi(image, [])

    context = smart_tool_context(crop=[{"x": 0, "y": 0}, {"x": 39, "y": 29}])
    context.pop("figure_id")
    call_handler(model, api, context, state={"settings": {}})

    assert model.predict_calls[0]["mode"] == "points"
    assert model.predict_calls[0]["bbox_coordinates"] == [0, 0, 30, 40]


def test_no_clicks_returns_the_existing_no_result_response(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [])

    _, result = call_handler(
        model, api, smart_tool_context(positive=[], negative=[]), state={"settings": {}}
    )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_click_outside_of_the_crop_returns_the_no_result_response(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [])

    context = smart_tool_context(crop=[{"x": 0, "y": 0}, {"x": 5, "y": 5}])
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_empty_prediction_returns_the_no_result_response(app_data_dir, image):
    empty_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    model = StubModel(image, empty_mask)
    api = FakeApi(image, [polygon_label(rect_ring(7, 5, 14, 10), figure_id=FIGURE_ID)])

    _, result = call_handler(
        model, api, smart_tool_context(init_figure=True), state={"settings": {}}
    )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert len(model.predict_calls) == 1
    assert os.listdir(app_data_dir) == []


def test_bad_request_returns_400(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, [])

    response, result = call_handler(model, api, {"image_id": IMAGE_ID}, state={"settings": {}})

    assert response.status_code == 400
    assert result == {"message": "400: Bad request.", "success": False}
    assert model.predict_calls == []
