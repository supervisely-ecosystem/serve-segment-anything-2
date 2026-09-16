"""Regressions for the production ``/smart_segmentation`` request flow.

The handler that ``src.main`` registers on the route is executed here directly;
only the external boundaries (Supervisely API, image cache and the model's
``predict``) are replaced, so request parsing, initial mask decoding, predictor
hand-off and bitmap response transport are the real code.
"""

import os
import sys

import numpy as np
import pytest
import supervisely as sly
from fastapi import Response

from src import init_mask
from src.smart_segmentation import smart_segmentation
from tests.support.smart_tool import (
    FIGURE_ID,
    IMAGE_HEIGHT,
    IMAGE_ID,
    IMAGE_WIDTH,
    LOCAL_FIGURE_ID,
    FakeApi,
    StubModel,
    bitmap_label,
    context_mask,
    decode_response_bitmap,
    encode_mask,
    make_image,
    make_request,
    smart_tool_context,
)


def call_handler(model, api, context, state=None):
    response = Response()
    result = smart_segmentation(model, response, make_request(context, state, api))
    return response, result


@pytest.fixture
def image():
    return make_image()


@pytest.fixture
def pred_mask():
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask[10:15, 20:24] = 255
    return mask


def init_figure_mask():
    """Tight mask with a hole and a disconnected part, as the tool rasterizes it."""
    data = np.zeros((10, 12), dtype=bool)
    data[0:6, 0:6] = True
    data[2:4, 2:4] = False
    data[8:10, 10:12] = True
    return data


def expected_init_mask(data, x, y):
    return (
        init_mask.place_on_image(data, (x, y), IMAGE_HEIGHT, IMAGE_WIDTH) * 255
    ).astype(np.uint8)


def test_handler_does_not_pull_in_the_model_stack():
    """The offline suite must not import/initialize the model stack."""
    assert "sam2" not in sys.modules
    assert "torch" not in sys.modules
    assert "src.main" not in sys.modules


# --------------------------------------------------------------------------- #
# direct mask -> predictor
# --------------------------------------------------------------------------- #
def test_direct_mask_without_figure_id_initializes_the_predictor(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    # any annotation download in this scenario is a failure
    api = FakeApi(image, fail_annotation_download=True)
    data = init_figure_mask()

    context = smart_tool_context(
        init_figure=True,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=context_mask(data, x=7, y=5),
    )
    response, result = call_handler(model, api, context, state={"settings": {}})

    assert response.status_code == 200
    assert api.annotation.calls == []  # no annotation API access at all
    assert api.image.get_info_calls == []
    assert len(model.predict_calls) == 1

    passed_mask = model.predict_calls[0]["init_mask"]
    assert passed_mask.dtype == np.uint8
    assert passed_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert passed_mask.tolist() == expected_init_mask(data, x=7, y=5).tolist()

    # cached for continuation under the platform's local figure id
    cached = model._init_mask_cache[LOCAL_FIGURE_ID]
    assert isinstance(cached, sly.Bitmap)
    assert (cached.origin.col, cached.origin.row) == (7, 5)

    # response transport is unchanged: origin + encoded bitmap of the prediction
    assert result["success"] is True and result["error"] is None
    assert result["origin"] == {"x": 20, "y": 10}
    assert decode_response_bitmap(result).tolist() == (pred_mask[10:15, 20:24] > 0).tolist()


@pytest.mark.parametrize(
    "x, y",
    [pytest.param(-4, -3, id="negative-origin"), pytest.param(55, 37, id="past-the-edge")],
)
def test_clipped_direct_mask_keeps_full_image_dimensions(
    app_data_dir, image, pred_mask, x, y
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)
    data = np.ones((8, 10), dtype=bool)

    context = smart_tool_context(
        init_figure=True, figure_id=FIGURE_ID, mask=context_mask(data, x=x, y=y)
    )
    _, result = call_handler(model, api, context, state={"settings": {}})

    passed_mask = model.predict_calls[0]["init_mask"]
    assert passed_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert passed_mask.tolist() == expected_init_mask(data, x=x, y=y).tolist()
    assert result["success"] is True


def test_mask_wins_over_a_supplied_figure_id(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    # the annotation holds a different figure; downloading it would be wrong
    api = FakeApi(
        image,
        [bitmap_label(np.ones((20, 20), dtype=bool), row=20, col=30)],
        fail_annotation_download=True,
    )
    data = np.ones((4, 6), dtype=bool)

    context = smart_tool_context(
        init_figure=True,
        figure_id=FIGURE_ID,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=context_mask(data, x=2, y=3),
    )
    response, result = call_handler(model, api, context, state={"settings": {}})

    assert response.status_code == 200 and result["success"] is True
    assert api.annotation.calls == []
    assert model.predict_calls[0]["init_mask"].tolist() == (
        expected_init_mask(data, x=2, y=3).tolist()
    )
    # continuation identity is the local figure id, not the legacy figure_id
    assert LOCAL_FIGURE_ID in model._init_mask_cache
    assert FIGURE_ID not in model._init_mask_cache


@pytest.mark.parametrize(
    "cache_field, cache_key",
    [
        pytest.param("local_figure_id", LOCAL_FIGURE_ID, id="local-figure-id"),
        pytest.param("figure_id", FIGURE_ID, id="legacy-figure-id"),
    ],
)
def test_continuation_click_reuses_the_cached_mask(
    app_data_dir, image, pred_mask, cache_field, cache_key
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)
    data = init_figure_mask()

    first_context = smart_tool_context(
        init_figure=True, mask=context_mask(data, x=7, y=5), **{cache_field: cache_key}
    )
    call_handler(model, api, first_context, state={"settings": {}})
    first_mask = model.predict_calls[0]["init_mask"]

    # a later click carries no mask at all
    second_context = smart_tool_context(
        positive=[{"x": 10, "y": 12}, {"x": 15, "y": 16}], **{cache_field: cache_key}
    )
    _, result = call_handler(model, api, second_context, state={"settings": {}})

    assert api.annotation.calls == []
    assert model.cache.download_image_calls == [IMAGE_ID]  # image cache reused
    assert len(model.predict_calls) == 2
    assert model.predict_calls[1]["init_mask"].tolist() == first_mask.tolist()
    assert result["success"] is True


def test_another_figures_cache_entry_is_not_reused(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)

    call_handler(
        model,
        api,
        smart_tool_context(
            init_figure=True,
            local_figure_id=LOCAL_FIGURE_ID,
            mask=context_mask(np.ones((4, 4), dtype=bool), x=1, y=1),
        ),
        state={"settings": {}},
    )
    _, result = call_handler(
        model,
        api,
        smart_tool_context(local_figure_id=LOCAL_FIGURE_ID + 1),
        state={"settings": {}},
    )

    assert model.predict_calls[1]["init_mask"] is None
    assert result["success"] is True


def test_mask_on_a_video_frame_is_placed_in_frame_coordinates(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)
    data = np.ones((5, 5), dtype=bool)

    context = {
        "video": {"video_id": 5, "frame_index": 3},
        "init_figure": True,
        "local_figure_id": LOCAL_FIGURE_ID,
        "mask": context_mask(data, x=12, y=9),
        "positive": [{"x": 13, "y": 10}],
        "negative": [],
    }
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert api.annotation.calls == []
    assert model.cache.download_frame_calls == [(5, 3)]
    assert model.predict_calls[0]["init_mask"].tolist() == (
        expected_init_mask(data, x=12, y=9).tolist()
    )
    assert model.predict_calls[0]["input_image_id"] == "5_3"
    assert result["success"] is True


def test_camel_case_context_fields_are_accepted(app_data_dir, image, pred_mask):
    """Legacy callers spell the contract fields in camelCase."""
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)
    data = np.ones((4, 6), dtype=bool)

    context = smart_tool_context(
        initFigure=True,
        figureId=FIGURE_ID,
        localFigureId=LOCAL_FIGURE_ID,
        mask=context_mask(data, x=2, y=3),
    )
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert api.annotation.calls == []
    assert model.predict_calls[0]["init_mask"].tolist() == (
        expected_init_mask(data, x=2, y=3).tolist()
    )
    # continuation identity comes from the camelCase local figure id
    assert LOCAL_FIGURE_ID in model._init_mask_cache
    assert FIGURE_ID not in model._init_mask_cache
    assert result["success"] is True


def test_camel_case_maskless_legacy_request_downloads_the_figure(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    data = np.ones((3, 5), dtype=bool)
    api = FakeApi(image, [bitmap_label(data, row=4, col=6, figure_id=FIGURE_ID)])

    context = smart_tool_context(initFigure=True, figureId=FIGURE_ID)
    response, result = call_handler(model, api, context, state={"settings": {}})

    assert response.status_code == 200 and result["success"] is True
    assert api.annotation.calls == [IMAGE_ID]
    assert model.predict_calls[0]["init_mask"].tolist() == (
        expected_init_mask(data, x=6, y=4).tolist()
    )


# --------------------------------------------------------------------------- #
# deprecated legacy figure-id path
# --------------------------------------------------------------------------- #
def test_maskless_legacy_request_still_downloads_the_figure(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    data = np.ones((6, 8), dtype=bool)
    api = FakeApi(image, [bitmap_label(data, row=5, col=7, figure_id=FIGURE_ID)])

    context = smart_tool_context(init_figure=True, figure_id=FIGURE_ID)
    response, result = call_handler(model, api, context, state={"settings": {}})

    assert response.status_code == 200 and result["success"] is True
    assert api.annotation.calls == [IMAGE_ID]
    assert model.predict_calls[0]["init_mask"].tolist() == (
        expected_init_mask(data, x=7, y=5).tolist()
    )
    assert isinstance(model._init_mask_cache[FIGURE_ID], sly.Bitmap)


def test_legacy_request_without_mask_and_without_figure_id_is_rejected(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image)

    response, result = call_handler(
        model, api, smart_tool_context(init_figure=True), state={"settings": {}}
    )

    assert response.status_code == 400
    assert result["success"] is False
    assert "without a usable 'figure_id'" in result["error"]
    assert api.annotation.calls == []
    assert model.predict_calls == []
    assert os.listdir(app_data_dir) == []


def test_request_without_any_initial_figure_has_no_init_mask(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)

    _, result = call_handler(model, api, smart_tool_context(), state={"settings": {}})

    assert api.annotation.calls == []
    assert model.predict_calls[0]["init_mask"] is None
    assert result["success"] is True


# --------------------------------------------------------------------------- #
# explicit errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mask_json, message",
    [
        pytest.param(
            {"data": encode_mask(np.ones((2, 2), dtype=bool))}, "origin", id="origin-missing"
        ),
        pytest.param(
            {"origin": [1, 2, 3], "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "pixel pair",
            id="origin-too-long",
        ),
        pytest.param({"origin": [1, 2], "data": "%%%not-base64%%%"}, "encoded bitmap", id="bad-data"),
        pytest.param({"origin": [1, 2], "data": ""}, "non-empty base64", id="empty-data"),
        pytest.param(
            context_mask(np.ones((3, 3), dtype=bool), x=IMAGE_WIDTH + 2, y=0),
            "no pixels inside",
            id="outside-the-image",
        ),
        pytest.param("mask", "must be an object", id="not-an-object"),
    ],
)
def test_malformed_mask_returns_an_explicit_error(
    app_data_dir, image, pred_mask, mask_json, message
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)

    context = smart_tool_context(
        init_figure=True, figure_id=FIGURE_ID, local_figure_id=LOCAL_FIGURE_ID, mask=mask_json
    )
    response, result = call_handler(model, api, context, state={"settings": {}})

    assert response.status_code == 400
    assert result["success"] is False
    assert message in result["error"]
    assert result["bitmap"] is None and result["origin"] is None
    # a malformed mask never falls back to the annotation download
    assert api.annotation.calls == []
    assert model.predict_calls == []
    assert LOCAL_FIGURE_ID not in model._init_mask_cache
    # the temporary image of the failed request is cleaned up
    assert os.listdir(app_data_dir) == []


def test_a_malformed_mask_does_not_reuse_the_cached_mask(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)

    call_handler(
        model,
        api,
        smart_tool_context(
            init_figure=True,
            local_figure_id=LOCAL_FIGURE_ID,
            mask=context_mask(np.ones((4, 4), dtype=bool), x=1, y=1),
        ),
        state={"settings": {}},
    )
    response, result = call_handler(
        model,
        api,
        smart_tool_context(local_figure_id=LOCAL_FIGURE_ID, mask={"origin": [1, 2]}),
        state={"settings": {}},
    )

    assert response.status_code == 400
    assert result["success"] is False
    assert len(model.predict_calls) == 1  # the second request did not predict


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
    api = FakeApi(image, fail_annotation_download=True)
    data = np.ones((6, 6), dtype=bool)

    context = smart_tool_context(
        init_figure=True,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=context_mask(data, x=21, y=11),
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
    assert settings["init_mask"].tolist() == expected_init_mask(data, x=21, y=11).tolist()

    assert result["origin"] == {"x": 22, "y": 12}
    assert decode_response_bitmap(result).tolist() == np.ones((4, 3), dtype=bool).tolist()


def test_predictor_receives_points_and_full_image_settings(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image, fail_annotation_download=True)

    context = smart_tool_context(
        init_figure=True,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=context_mask(np.ones((4, 4), dtype=bool), x=8, y=6),
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


def test_points_mode_when_bbox_switch_is_off(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask, use_bbox=False)
    api = FakeApi(image)

    context = smart_tool_context(crop=[{"x": 0, "y": 0}, {"x": 39, "y": 29}])
    call_handler(model, api, context, state={"settings": {}})

    assert model.predict_calls[0]["mode"] == "points"
    assert model.predict_calls[0]["bbox_coordinates"] == [0, 0, 30, 40]


def test_no_clicks_returns_the_existing_no_result_response(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image)

    _, result = call_handler(
        model, api, smart_tool_context(positive=[], negative=[]), state={"settings": {}}
    )

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_click_outside_of_the_crop_returns_the_no_result_response(
    app_data_dir, image, pred_mask
):
    model = StubModel(image, pred_mask)
    api = FakeApi(image)

    context = smart_tool_context(crop=[{"x": 0, "y": 0}, {"x": 5, "y": 5}])
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_empty_prediction_returns_the_no_result_response(app_data_dir, image):
    empty_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    model = StubModel(image, empty_mask)
    api = FakeApi(image, fail_annotation_download=True)

    context = smart_tool_context(
        init_figure=True, mask=context_mask(np.ones((4, 4), dtype=bool), x=1, y=1)
    )
    _, result = call_handler(model, api, context, state={"settings": {}})

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert len(model.predict_calls) == 1
    assert os.listdir(app_data_dir) == []


def test_bad_request_returns_400(app_data_dir, image, pred_mask):
    model = StubModel(image, pred_mask)
    api = FakeApi(image)

    response, result = call_handler(model, api, {"image_id": IMAGE_ID}, state={"settings": {}})

    assert response.status_code == 400
    assert result == {"message": "400: Bad request.", "success": False}
    assert model.predict_calls == []
