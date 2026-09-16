#!/usr/bin/env python3
"""Reproduces the issue on the exact baseline commit (no GPU, no instance).

Runs the baseline ``/smart_segmentation`` route (read from git, see
``master_route.py``) with the shared direct-mask contract payload and records
what the baseline actually does:

1. a contract mask without ``figure_id`` cannot initialize the predictor: the
   baseline still downloads the annotation and the request fails;
2. when both a mask and a ``figure_id`` are present the mask is ignored and the
   downloaded figure is used instead;
3. a mask without ``init_figure`` is dropped silently: the predictor gets no
   initial mask at all.

Exits 0 when the baseline behaves as described (issue reproduced) and non-zero
if the baseline already implements the contract.
"""

import logging
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("SLY_APP_DATA_DIR", tempfile.mkdtemp(prefix="baseline-app-data-"))

import numpy as np  # noqa: E402
from fastapi import Response  # noqa: E402
from supervisely.sly_logger import logger as sly_logger  # noqa: E402

# keep the evidence readable: the route's warn logs carry full tracebacks
sly_logger.setLevel(logging.ERROR)

from tests.replay.master_route import BASELINE_SHA, load_baseline_route  # noqa: E402
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
    make_image,
    make_request,
    smart_tool_context,
)

MASK_DATA = np.ones((6, 8), dtype=bool)
MASK_ORIGIN = (7, 5)  # x, y
DECOY = np.ones((4, 4), dtype=bool)
DECOY_ORIGIN = (30, 20)  # col, row


def run(route, api, context):
    image, pred_mask = api.image.image, np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
    pred_mask[10:15, 20:24] = 255
    model = StubModel(image, pred_mask)
    response = Response()
    try:
        result = route(model, response, make_request(context, {"settings": {}}, api))
        return model, response, result, None
    except BaseException as error:  # the baseline route does not handle this
        return model, response, None, error


def main():
    route, verified = load_baseline_route()
    image = make_image()
    mask = context_mask(MASK_DATA, x=MASK_ORIGIN[0], y=MASK_ORIGIN[1])
    source = "read from git and byte-identical to the committed fixture" if verified else (
        "committed fixture (the commit is not readable in this checkout)"
    )
    print(f"baseline commit: {BASELINE_SHA} ({source})")

    # 1. contract mask, no figure_id
    api = FakeApi(image)
    model, _, result, error = run(
        route,
        api,
        smart_tool_context(init_figure=True, local_figure_id=LOCAL_FIGURE_ID, mask=mask),
    )
    assert error is not None, f"baseline unexpectedly served a maskless-id request: {result}"
    assert model.predict_calls == [], "baseline unexpectedly reached the predictor"
    assert api.annotation.calls == [IMAGE_ID], api.annotation.calls
    print(
        "1. mask without figure_id -> "
        f"annotation download attempted for image {api.annotation.calls[0]}, "
        f"request failed with {type(error).__name__}: {error}"
    )

    # 2. mask and figure_id: which one wins?
    api = FakeApi(
        image, [bitmap_label(DECOY, row=DECOY_ORIGIN[1], col=DECOY_ORIGIN[0], figure_id=FIGURE_ID)]
    )
    model, _, result, error = run(
        route,
        api,
        smart_tool_context(init_figure=True, figure_id=FIGURE_ID, mask=mask),
    )
    assert error is None, error
    used = model.predict_calls[0]["init_mask"]
    assert used is not None and used.any(), "baseline produced no initial mask"
    contract_pixels = set(zip(*np.nonzero(used[5:11, 7:15])))
    assert not contract_pixels, "baseline unexpectedly used the contract mask"
    rows, cols = np.nonzero(used)
    print(
        "2. mask and figure_id -> baseline ignored the mask at origin "
        f"{list(MASK_ORIGIN)} and used the downloaded figure at "
        f"(x={cols.min()}, y={rows.min()}); annotation downloads: {len(api.annotation.calls)}"
    )

    # 3. mask without init_figure
    api = FakeApi(image)
    model, _, result, error = run(
        route, api, smart_tool_context(local_figure_id=LOCAL_FIGURE_ID, mask=mask)
    )
    assert error is None, error
    assert model.predict_calls[0]["init_mask"] is None, "baseline used the mask"
    assert result["success"] is True
    print(
        "3. mask without init_figure -> silently ignored, "
        f"predictor init_mask={model.predict_calls[0]['init_mask']}, "
        f"response success={result['success']}"
    )

    print(
        f"REPRODUCED at {BASELINE_SHA[:7]}: the direct-mask contract is not implemented "
        f"({IMAGE_HEIGHT}x{IMAGE_WIDTH} image, tight mask {MASK_DATA.shape} at "
        f"origin {list(MASK_ORIGIN)})"
    )


if __name__ == "__main__":
    main()
