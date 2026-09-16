"""Offline stand-ins for the boundaries of the Smart Tool request flow.

Only the external boundaries are faked here (Supervisely API, the image cache
and the model's ``predict``), so request parsing, initial mask decoding,
predictor hand-off and bitmap response transport stay production code.

The module is shared by the approved ``tests/unit`` suite and by the replay
scripts in ``tests/replay``, which run the same scenarios against the baseline
and the current source without pytest.
"""

import os
import threading
from types import SimpleNamespace

import numpy as np
import supervisely as sly
from cacheout import Cache
from cachetools import LRUCache

IMAGE_ID = 777
FIGURE_ID = 4242
LOCAL_FIGURE_ID = 90210
IMAGE_HEIGHT, IMAGE_WIDTH = 40, 60


class AnnotationDownloadCalled(AssertionError):
    """Sentinel: the request reached the annotation API although it must not."""


def make_image(height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    return image


def encode_mask(data: np.ndarray) -> str:
    """Encodes a tight mask exactly like Supervisely ``Bitmap`` JSON does."""
    return sly.Bitmap.data_2_base64(np.asarray(data).astype(bool))


def context_mask(data: np.ndarray, x: int, y: int) -> dict:
    """Builds the contract mask: tight bitmap + its top-left image position."""
    return {"origin": [x, y], "data": encode_mask(data)}


def bitmap_label(data: np.ndarray, row: int, col: int, figure_id: int = FIGURE_ID) -> dict:
    """Builds a bitmap label the way ``api.annotation.download_json`` returns it."""
    bitmap = sly.Bitmap(
        np.asarray(data).astype(bool), origin=sly.PointLocation(row=row, col=col)
    )
    return {"id": figure_id, "classTitle": "mask", **bitmap.to_json()}


class FakeImageApi:
    def __init__(self, image):
        self.image = image
        self.download_np_calls = []
        self.get_info_calls = []

    def download_np(self, image_id):
        self.download_np_calls.append(image_id)
        return self.image

    def get_info_by_id(self, image_id):
        self.get_info_calls.append(image_id)
        return SimpleNamespace(
            id=image_id, height=self.image.shape[0], width=self.image.shape[1]
        )


class FakeAnnotationApi:
    def __init__(self, objects=(), fail=False):
        self.objects = list(objects)
        self.fail = fail
        self.calls = []

    def download_json(self, image_id):
        self.calls.append(image_id)
        if self.fail:
            raise AnnotationDownloadCalled(
                f"annotation of image {image_id} must not be downloaded"
            )
        return {"objects": list(self.objects)}


class FakeApi:
    """Only the endpoints the Smart Tool flow is allowed to use exist here."""

    def __init__(self, image, objects=(), fail_annotation_download=False):
        self.image = FakeImageApi(image)
        self.annotation = FakeAnnotationApi(objects, fail=fail_annotation_download)


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


def make_request(context, state=None, api=None):
    return SimpleNamespace(
        state=SimpleNamespace(context=context, state=state or {}, api=api)
    )


def smart_tool_context(**overrides):
    """Image Smart Tool context with one positive click."""
    context = {
        "image_id": IMAGE_ID,
        "positive": [{"x": 10, "y": 12}],
        "negative": [],
    }
    context.update(overrides)
    return context


def decode_response_bitmap(result) -> np.ndarray:
    return sly.Bitmap.base64_2_data(result["bitmap"])
