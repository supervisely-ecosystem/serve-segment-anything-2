"""The init mask of /smart_segmentation, offline.

The route is called directly with a pre-seeded image cache and a stubbed predictor,
so nothing here loads SAM 2 weights, needs a GPU or talks to a platform.
"""

import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cacheout import Cache
from cachetools import LRUCache
from fastapi import Response

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# src/main.py builds its GUI and registers its routes at import time. TASK_ID stays
# unset: with it the app asks the platform for its task while importing.
os.environ.setdefault("SERVER_ADDRESS", "http://localhost:8000")
os.environ.setdefault("API_TOKEN", "0" * 32)
os.environ.setdefault("TEAM_ID", "1")
os.environ.setdefault("WORKSPACE_ID", "1")
os.environ.setdefault("SLY_APP_DATA_DIR", tempfile.mkdtemp(prefix="sam2-init-mask-"))

import supervisely as sly  # noqa: E402

import src.main as app_main  # noqa: E402

IMAGE_ID = 777
FIGURE_ID = 4242
IMAGE_HEIGHT, IMAGE_WIDTH = 60, 80


def _polygon_figure() -> sly.Polygon:
    """A polygon with a hole, so a mask built from the wrong field cannot match by luck."""
    corners = lambda l, t, r, b: [  # noqa: E731
        sly.PointLocation(row=t, col=l),
        sly.PointLocation(row=t, col=r),
        sly.PointLocation(row=b, col=r),
        sly.PointLocation(row=b, col=l),
    ]
    return sly.Polygon(exterior=corners(20, 10, 40, 30), interior=[corners(28, 18, 32, 22)])


def _context(**overrides) -> dict:
    context = {
        "image_id": IMAGE_ID,
        "positive": [{"x": 24, "y": 12}],
        "negative": [],
        "crop": [{"x": 15, "y": 5}, {"x": 45, "y": 35}],
        "figure_id": FIGURE_ID,
    }
    context.update(overrides)
    return context


class _ForbiddenApi:
    """Fails the test as soon as the route reaches for the platform."""

    def __getattr__(self, name):
        raise AssertionError(f"the inline mask path must not use the API (api.{name})")


@pytest.fixture
def serve_request(monkeypatch):
    model = app_main.m
    route = {
        r.path: r.endpoint for r in model.app.get_server().routes if hasattr(r, "endpoint")
    }["/smart_segmentation"]

    # load_on_device() builds the smart tool state together with the weights; only the
    # state is needed here, and loading SAM 2 would need a GPU. The route downloads the
    # image only when it is not cached, so a seeded cache keeps the request offline
    # without stubbing the download itself.
    image_cache = Cache(ttl=60)
    image_cache.set(str(IMAGE_ID), np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, np.uint8))
    monkeypatch.setattr(model, "_inference_image_cache", image_cache, raising=False)
    monkeypatch.setattr(model, "_init_mask_cache", LRUCache(maxsize=100), raising=False)
    monkeypatch.setattr(model, "_inference_image_lock", threading.Lock(), raising=False)

    predicted = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), bool)
    predicted[10:20, 20:30] = True
    seen = {}

    def fake_predict(image_path, settings):
        seen["settings"] = dict(settings)
        return [SimpleNamespace(mask=predicted)]

    monkeypatch.setattr(model, "predict", fake_predict)

    def run(context, api=None):
        request = SimpleNamespace(
            state=SimpleNamespace(context=context, state={}, api=api or _ForbiddenApi())
        )
        payload = route(response=Response(), request=request)
        return payload, seen.get("settings")

    return run


def test_a_polygon_figure_is_handed_back_to_the_model_as_its_mask(serve_request):
    polygon = _polygon_figure()
    context = _context(
        mask={"geometry_type": polygon.geometry_name(), "geometry": polygon.to_json()}
    )

    payload, settings = serve_request(context)

    assert payload["success"] is True
    init_mask = settings["init_mask"]
    assert init_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert set(np.unique(init_mask)) == {0, 255}
    assert init_mask[15, 25] == 255, "inside the polygon"
    assert init_mask[20, 30] == 0, "the polygon's hole"
    assert init_mask[50, 70] == 0, "outside the polygon"


def test_the_figure_is_kept_for_the_clicks_that_follow(serve_request):
    polygon = _polygon_figure()
    first_click = _context(
        mask={"geometry_type": polygon.geometry_name(), "geometry": polygon.to_json()}
    )

    _, first_settings = serve_request(first_click)
    # The Smart Tool sends the figure with the first request of a session only.
    _, next_settings = serve_request(_context())

    assert first_settings["init_mask"] is not None
    np.testing.assert_array_equal(next_settings["init_mask"], first_settings["init_mask"])
