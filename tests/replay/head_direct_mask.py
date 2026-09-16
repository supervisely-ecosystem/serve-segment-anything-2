#!/usr/bin/env python3
"""Checks the direct-mask contract on the current source (no GPU, no instance).

Runs the production ``/smart_segmentation`` handler through the same offline
stand-ins the baseline reproduction uses and asserts, with exact full-image
arrays, that the shared contract is implemented: mask without ``figure_id``,
mask precedence over ``figure_id`` (with a failing annotation download),
continuation reuse, clipping, explicit malformed-mask errors and the deprecated
legacy figure-id path. Exits non-zero on any regression.
"""

import logging
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("SLY_APP_DATA_DIR", tempfile.mkdtemp(prefix="head-app-data-"))

import numpy as np  # noqa: E402
from fastapi import Response  # noqa: E402
from supervisely.sly_logger import logger as sly_logger  # noqa: E402

# keep the evidence readable: the handler's own warn logs carry full tracebacks
sly_logger.setLevel(logging.ERROR)

from src import init_mask  # noqa: E402
from src.smart_segmentation import smart_segmentation  # noqa: E402
from tests.support.smart_tool import (  # noqa: E402
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
    make_image,
    make_request,
    smart_tool_context,
)

MASK_DATA = np.ones((6, 8), dtype=bool)
MASK_DATA[1:3, 1:3] = False  # hole, must survive the transport
MASK_ORIGIN = (7, 5)  # x, y
DECOY = np.ones((4, 4), dtype=bool)
IMAGE = make_image()
PRED_MASK = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
PRED_MASK[10:15, 20:24] = 255


def expected(data, x, y):
    return (
        init_mask.place_on_image(data, (x, y), IMAGE_HEIGHT, IMAGE_WIDTH) * 255
    ).astype(np.uint8)


def call(model, api, context):
    response = Response()
    result = smart_segmentation(model, response, make_request(context, {"settings": {}}, api))
    return response, result


def new_model():
    return StubModel(IMAGE, PRED_MASK)


def main():
    mask = context_mask(MASK_DATA, x=MASK_ORIGIN[0], y=MASK_ORIGIN[1])

    # 1. contract mask without figure_id, no annotation access allowed
    model, api = new_model(), FakeApi(IMAGE, fail_annotation_download=True)
    response, result = call(
        model,
        api,
        smart_tool_context(init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask),
    )
    assert response.status_code == 200 and result["success"] is True, result
    assert len(model.predict_calls) == 1, f"predictor not reached: {result}"
    used = model.predict_calls[0]["init_mask"]
    assert api.annotation.calls == [] and api.image.get_info_calls == []
    assert used.dtype == np.uint8 and used.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert used.tolist() == expected(MASK_DATA, *MASK_ORIGIN).tolist()
    assert result["origin"] == {"x": 20, "y": 10}
    assert decode_response_bitmap(result).tolist() == (PRED_MASK[10:15, 20:24] > 0).tolist()
    print(
        "1. mask without figure_id -> no annotation download, predictor got the exact "
        f"{used.shape} mask at origin {list(MASK_ORIGIN)} "
        f"({int((used > 0).sum())} px, hole preserved: {used[6, 8] == 0}); "
        f"response origin {result['origin']}"
    )

    # 2. mask and figure_id: the mask wins and nothing is downloaded
    model, api = new_model(), FakeApi(
        IMAGE, [bitmap_label(DECOY, row=20, col=30, figure_id=FIGURE_ID)],
        fail_annotation_download=True,
    )
    _, result = call(
        model,
        api,
        smart_tool_context(
            init_figure=True, figure_id=FIGURE_ID, local_figure_id=LOCAL_FIGURE_ID, mask=mask
        ),
    )
    assert result["success"] is True and api.annotation.calls == []
    assert len(model.predict_calls) == 1, f"predictor not reached: {result}"
    used = model.predict_calls[0]["init_mask"]
    assert used.tolist() == expected(MASK_DATA, *MASK_ORIGIN).tolist()
    assert LOCAL_FIGURE_ID in model._init_mask_cache and FIGURE_ID not in model._init_mask_cache
    print(
        "2. mask and figure_id -> mask wins, annotation downloads: "
        f"{len(api.annotation.calls)}, cache key: local_figure_id={LOCAL_FIGURE_ID}"
    )

    # 3. continuation click without a mask reuses the cached mask
    _, result = call(
        model,
        api,
        smart_tool_context(
            local_figure_id=LOCAL_FIGURE_ID, positive=[{"x": 10, "y": 12}, {"x": 15, "y": 16}]
        ),
    )
    assert result["success"] is True and api.annotation.calls == []
    assert model.predict_calls[1]["init_mask"].tolist() == used.tolist()
    assert model.cache.download_image_calls == [IMAGE_ID]
    print(
        "3. continuation click without mask -> identical init mask reused from cache, "
        f"image downloads: {len(model.cache.download_image_calls)}"
    )

    # 4. clipping at the image bounds
    for x, y in ((-4, -3), (IMAGE_WIDTH - 3, IMAGE_HEIGHT - 2)):
        model, api = new_model(), FakeApi(IMAGE, fail_annotation_download=True)
        _, result = call(
            model,
            api,
            smart_tool_context(init_figure=True, mask=context_mask(np.ones((8, 10), bool), x, y)),
        )
        clipped = model.predict_calls[0]["init_mask"]
        assert result["success"] is True
        assert clipped.tolist() == expected(np.ones((8, 10), bool), x, y).tolist()
        print(
            f"4. mask at origin [{x}, {y}] -> clipped to the image, "
            f"{int((clipped > 0).sum())} px inside, shape {clipped.shape}"
        )

    # 5. malformed masks are explicit handled errors, never a silent fallback
    for label, bad_mask in (
        ("undecodable data", {"origin": [1, 2], "data": "%%%bad%%%"}),
        ("non-finite origin", {"origin": [float("nan"), 2], "data": context_mask(MASK_DATA, 0, 0)["data"]}),
        ("origin outside the image", context_mask(MASK_DATA, IMAGE_WIDTH + 4, 0)),
    ):
        model, api = new_model(), FakeApi(IMAGE, fail_annotation_download=True)
        response, result = call(
            model,
            api,
            smart_tool_context(init_figure=True, figure_id=FIGURE_ID, mask=bad_mask),
        )
        assert response.status_code == 400 and result["success"] is False, result
        assert result["bitmap"] is None and result["origin"] is None, result
        assert model.predict_calls == [] and api.annotation.calls == []
        assert os.listdir(os.environ["SLY_APP_DATA_DIR"]) == []
        print(
            f"5. malformed mask ({label}) -> HTTP {response.status_code}, "
            f"no predict, no download, temp files left: "
            f"{len(os.listdir(os.environ['SLY_APP_DATA_DIR']))}, error: {result['error']}"
        )

    # 6. deprecated legacy path: no mask, figure downloaded by id
    model, api = new_model(), FakeApi(
        IMAGE, [bitmap_label(DECOY, row=20, col=30, figure_id=FIGURE_ID)]
    )
    _, result = call(model, api, smart_tool_context(init_figure=True, figure_id=FIGURE_ID))
    legacy = model.predict_calls[0]["init_mask"]
    assert result["success"] is True and api.annotation.calls == [IMAGE_ID]
    assert legacy.tolist() == expected(DECOY, 30, 20).tolist()
    assert FIGURE_ID in model._init_mask_cache
    print(
        "6. legacy maskless figure_id -> still initialized from the downloaded figure at "
        f"(x=30, y=20), {int((legacy > 0).sum())} px, annotation downloads: "
        f"{len(api.annotation.calls)}"
    )

    print("HEAD OK: direct-mask contract served without annotation access")


if __name__ == "__main__":
    main()
