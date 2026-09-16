"""Reproduction of the issue on the baseline commit, inside the approved suite.

The same extracted baseline route the replay script uses (see
``tests/replay/master_route.py``) is executed here against the offline
stand-ins, so the approved suite itself records *why* the change is needed:
before it, a direct contract mask cannot initialize the predictor.

If someone ever claims the baseline already served the contract, these tests
fail; they also fail if the committed baseline fixture drifts from the commit
it names.
"""

import ast

import numpy as np
import pytest
from fastapi import Response

from tests.replay.master_route import BASELINE_SHA, baseline_route_source, load_baseline_route
from tests.support.smart_tool import (
    FIGURE_ID,
    IMAGE_ID,
    LOCAL_FIGURE_ID,
    AnnotationDownloadCalled,
    FakeApi,
    StubModel,
    bitmap_label,
    context_mask,
    make_image,
    make_request,
    smart_tool_context,
)

MASK_DATA = np.ones((6, 8), dtype=bool)
MASK_X, MASK_Y = 7, 5
DECOY_ROW, DECOY_COL = 20, 30


@pytest.fixture(scope="module")
def baseline_route():
    route, _ = load_baseline_route()
    return route


def call_baseline(route, model, api, context):
    response = Response()
    return response, route(model, response, make_request(context, {"settings": {}}, api))


def test_the_baseline_fixture_is_the_route_of_the_named_commit():
    """The reproduction must replay real history, not a hand-written baseline."""
    source, verified_against_git = baseline_route_source()

    assert "download_init_mask(api, figure_id, image_id)" in source
    literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # the baseline reads the legacy fields only; the contract mask is unknown to it
    assert {"figure_id", "init_figure"} <= literals
    assert "mask" not in literals and "local_figure_id" not in literals
    if not verified_against_git:
        pytest.skip(f"baseline commit {BASELINE_SHA} is not readable in this checkout")


def test_baseline_cannot_initialize_from_a_mask_without_figure_id(
    app_data_dir, baseline_route
):
    image = make_image()
    model = StubModel(image, np.zeros(image.shape[:2], dtype=np.uint8))
    api = FakeApi(image)

    context = smart_tool_context(
        init_figure=True,
        local_figure_id=LOCAL_FIGURE_ID,
        mask=context_mask(MASK_DATA, x=MASK_X, y=MASK_Y),
    )
    with pytest.raises(AssertionError) as failure:
        call_baseline(baseline_route, model, api, context)

    # the baseline goes to the annotation API and fails on the missing figure
    assert api.annotation.calls == [IMAGE_ID]
    assert "not found" in str(failure.value)
    assert model.predict_calls == []


def test_baseline_ignores_the_mask_when_a_figure_id_is_also_present(
    app_data_dir, baseline_route
):
    image = make_image()
    pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pred_mask[10:15, 20:24] = 255
    model = StubModel(image, pred_mask)
    api = FakeApi(
        image,
        [bitmap_label(np.ones((4, 4), bool), row=DECOY_ROW, col=DECOY_COL, figure_id=FIGURE_ID)],
    )

    context = smart_tool_context(
        init_figure=True,
        figure_id=FIGURE_ID,
        mask=context_mask(MASK_DATA, x=MASK_X, y=MASK_Y),
    )
    _, result = call_baseline(baseline_route, model, api, context)

    assert result["success"] is True
    assert api.annotation.calls == [IMAGE_ID]  # downloaded instead of using the mask
    used = model.predict_calls[0]["init_mask"]
    rows, cols = np.nonzero(used)
    # the predictor got the downloaded figure, not the mask sent in the request
    assert (cols.min(), rows.min()) == (DECOY_COL, DECOY_ROW)
    assert not used[
        MASK_Y : MASK_Y + MASK_DATA.shape[0], MASK_X : MASK_X + MASK_DATA.shape[1]
    ].any()


def test_baseline_drops_a_mask_sent_without_init_figure(app_data_dir, baseline_route):
    image = make_image()
    model = StubModel(image, np.ones(image.shape[:2], dtype=np.uint8) * 255)
    api = FakeApi(image, fail_annotation_download=True)

    context = smart_tool_context(
        local_figure_id=LOCAL_FIGURE_ID, mask=context_mask(MASK_DATA, x=MASK_X, y=MASK_Y)
    )
    _, result = call_baseline(baseline_route, model, api, context)

    assert result["success"] is True
    assert model.predict_calls[0]["init_mask"] is None  # silently ignored
    assert api.annotation.calls == []


def test_the_annotation_download_sentinel_would_catch_a_baseline_download(
    app_data_dir, baseline_route
):
    """Guards the head tests: the sentinel API really fails the request."""
    image = make_image()
    model = StubModel(image, np.zeros(image.shape[:2], dtype=np.uint8))
    api = FakeApi(image, fail_annotation_download=True)

    with pytest.raises(AnnotationDownloadCalled):
        call_baseline(
            baseline_route,
            model,
            api,
            smart_tool_context(init_figure=True, figure_id=FIGURE_ID),
        )
