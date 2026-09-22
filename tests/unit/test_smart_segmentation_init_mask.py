"""Offline checks for the init mask of the app's /smart_segmentation route.

The route is driven directly with a pre-seeded image cache and a stubbed
predictor, so these are CPU-only geometry and transport checks: no SAM 2 weights
are loaded, no inference runs and no server is contacted.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The app module builds its GUI and registers its routes at import time. TASK_ID
# is deliberately left unset: with it the app asks the platform for its task on
# import, and there is no platform here.
os.environ.setdefault("SERVER_ADDRESS", "http://localhost:8000")
os.environ.setdefault("API_TOKEN", "0" * 32)
os.environ.setdefault("TEAM_ID", "1")
os.environ.setdefault("WORKSPACE_ID", "1")
os.environ.setdefault("SLY_APP_DATA_DIR", tempfile.mkdtemp(prefix="sam2-init-mask-"))

import supervisely as sly  # noqa: E402
from fastapi import Response  # noqa: E402

import src.main as app_main  # noqa: E402

IMAGE_ID = 777
FIGURE_ID = 4242
IMAGE_HEIGHT, IMAGE_WIDTH = 60, 80
FIGURE_ORIGIN_ROW, FIGURE_ORIGIN_COL = 7, 19


def _figure_bitmap() -> sly.Bitmap:
    """A bitmap figure with a hole, so a misaligned mask cannot match by luck."""
    data = np.zeros((11, 13), bool)
    data[1:10, 2:11] = True
    data[4:7, 5:8] = False
    return sly.Bitmap(
        data, origin=sly.PointLocation(row=FIGURE_ORIGIN_ROW, col=FIGURE_ORIGIN_COL)
    )


def _annotation_json() -> dict:
    return {"objects": [{**_figure_bitmap().to_json(), "id": FIGURE_ID}]}


def _mask_field() -> dict:
    """The figure's mask as the platform forwards it into the request context."""
    bitmap_json = _figure_bitmap().to_json()["bitmap"]
    origin_col, origin_row = bitmap_json["origin"]
    return {"data": bitmap_json["data"], "origin": {"x": origin_col, "y": origin_row}}


def _context(**overrides) -> dict:
    context = {
        "image_id": IMAGE_ID,
        "positive": [{"x": 24, "y": 12}],
        "negative": [],
        "crop": [{"x": 15, "y": 5}, {"x": 45, "y": 25}],
    }
    context.update(overrides)
    return context


class _RecordingApi:
    """Serves the deprecated figure_id path and records what it was asked for."""

    def __init__(self):
        self.calls = []
        self.annotation = SimpleNamespace(download_json=self._download_json)
        self.image = SimpleNamespace(get_info_by_id=self._get_info_by_id)

    def _download_json(self, image_id):
        self.calls.append(("annotation.download_json", image_id))
        return _annotation_json()

    def _get_info_by_id(self, image_id):
        self.calls.append(("image.get_info_by_id", image_id))
        return SimpleNamespace(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)


class _ForbiddenApi:
    """Fails the test as soon as the route reaches for the API."""

    def __getattr__(self, name):
        raise AssertionError(f"the inline mask path must not use the API (api.{name})")


@pytest.fixture(scope="module")
def routes():
    server = app_main.m.app.get_server()
    return {r.path: r.endpoint for r in server.routes if hasattr(r, "endpoint")}


@pytest.fixture
def serve_request(routes, monkeypatch):
    route = routes["/smart_segmentation"]
    model = app_main.m
    # load_on_device() sets the smart tool state up together with the weights;
    # only the state is needed here, and loading SAM 2 would need a GPU.
    monkeypatch.setattr(model, "_inference_image_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(model, "_init_mask_cache", LRUCache(maxsize=100), raising=False)
    # The route downloads the image only when it is not cached; caching it keeps
    # the request offline without stubbing the download itself.
    image_cache = Cache(ttl=60)
    image_cache.set(
        str(IMAGE_ID), np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, np.uint8)
    )
    monkeypatch.setattr(model, "_inference_image_cache", image_cache, raising=False)

    predicted = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), bool)
    predicted[10:20, 20:30] = True
    seen = {}

    def fake_predict(image_path, settings):
        seen["settings"] = dict(settings)
        return [SimpleNamespace(mask=predicted)]

    monkeypatch.setattr(model, "predict", fake_predict)

    def run(context, api):
        seen.clear()
        response = Response()
        request = SimpleNamespace(
            state=SimpleNamespace(context=context, state={}, api=api)
        )
        payload = route(response=response, request=request)
        return response, payload, seen.get("settings")

    return run


def test_inline_mask_is_served_without_api_and_matches_the_figure_download(
    serve_request,
):
    """`mask` yields the array the figure_id path yields, with no API access."""
    api = _RecordingApi()
    _, legacy_payload, legacy_settings = serve_request(
        _context(figure_id=FIGURE_ID, init_figure=True), api
    )
    assert legacy_payload["success"] is True
    assert [call for call, _ in api.calls] == [
        "annotation.download_json",
        "image.get_info_by_id",
    ]
    legacy_mask = legacy_settings["init_mask"]
    assert legacy_mask is not None

    _, inline_payload, inline_settings = serve_request(
        _context(mask=_mask_field(), init_figure=True), _ForbiddenApi()
    )
    assert inline_payload["success"] is True
    inline_mask = inline_settings["init_mask"]
    assert inline_mask is not None
    assert inline_mask.shape == legacy_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert inline_mask.dtype == legacy_mask.dtype == np.uint8
    assert set(np.unique(inline_mask)) == {0, 255}
    np.testing.assert_array_equal(inline_mask, legacy_mask)


def test_legacy_request_without_mask_still_uses_the_download_then_the_cache(
    serve_request,
):
    api = _RecordingApi()
    _, _, first = serve_request(_context(figure_id=FIGURE_ID, init_figure=True), api)
    _, _, repeat = serve_request(_context(figure_id=FIGURE_ID), api)
    assert [call for call, _ in api.calls] == [
        "annotation.download_json",
        "image.get_info_by_id",
        # The repeat request reuses the cached figure and only asks for the size.
        "image.get_info_by_id",
    ]
    np.testing.assert_array_equal(repeat["init_mask"], first["init_mask"])


@pytest.mark.parametrize(
    "mask_field",
    [
        pytest.param(
            {"data": "not a bitmap", "origin": {"x": 19, "y": 7}}, id="undecodable-data"
        ),
        pytest.param(
            {"data": _mask_field()["data"], "origin": [19, 7]}, id="origin-not-a-point"
        ),
    ],
)
def test_unusable_mask_is_a_bad_request_and_never_falls_back(serve_request, mask_field):
    api = _RecordingApi()
    response, payload, settings = serve_request(
        _context(figure_id=FIGURE_ID, init_figure=True, mask=mask_field), api
    )
    assert response.status_code == 400
    assert payload["success"] is False
    assert api.calls == []
    assert settings is None, "a bad request must not reach the predictor"


def test_batch_wrapper_forwards_the_mask_to_the_route(routes, monkeypatch):
    """The batch wrappers re-enter /smart_segmentation with the context as sent."""
    forwarded = []
    monkeypatch.setenv("TASK_ID", "1")

    def fake_send_request(session_id, endpoint, data, context, **kwargs):
        forwarded.append((endpoint, context))
        return {"success": True}

    monkeypatch.setattr(
        app_main.api, "task", SimpleNamespace(send_request=fake_send_request)
    )
    context = _context(mask=_mask_field())
    request = SimpleNamespace(
        state=SimpleNamespace(context={"states": [context]}, state={})
    )
    routes["/smart_segmentation_batch"](request=request)

    assert forwarded == [("smart_segmentation", context)]
