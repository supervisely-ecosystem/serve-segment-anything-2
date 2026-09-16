"""Contract tests for decoding the Smart Tool direct mask.

Everything here runs on CPU with plain numpy arrays: no weights, no model
initialization, no GPU and no Supervisely instance.
"""

import numpy as np
import pytest
import supervisely as sly

from src import init_mask
from tests.support.smart_tool import bitmap_label, context_mask, encode_mask

IMAGE_HEIGHT, IMAGE_WIDTH = 40, 60
IMAGE_ID = 777
FIGURE_ID = 4242


def tight_mask_with_hole_and_gap():
    """Tight 10x12 mask: a ring with a hole plus a disconnected block."""
    data = np.zeros((10, 12), dtype=bool)
    data[0:6, 0:6] = True
    data[2:4, 2:4] = False  # hole
    data[8:10, 10:12] = True  # disconnected part
    return data


def decoded_full_mask(mask_json, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    bitmap = init_mask.decode_context_mask(mask_json, height, width)
    return init_mask.bitmap_to_mask(bitmap, height, width)


def test_mask_is_placed_at_its_origin_with_holes_and_gaps_preserved():
    data = tight_mask_with_hole_and_gap()

    full_mask = decoded_full_mask(context_mask(data, x=7, y=5))

    expected = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected[5:15, 7:19] = data.astype(np.uint8) * 255
    assert full_mask.dtype == np.uint8
    assert full_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert full_mask.tolist() == expected.tolist()


def test_mask_at_the_image_origin_keeps_full_image_dimensions():
    data = np.ones((3, 4), dtype=bool)

    full_mask = decoded_full_mask(context_mask(data, x=0, y=0))

    expected = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected[0:3, 0:4] = 255
    assert full_mask.tolist() == expected.tolist()


@pytest.mark.parametrize(
    "x, y, rows, cols",
    [
        pytest.param(-4, -3, slice(0, 3), slice(0, 4), id="top-left-overflow"),
        pytest.param(IMAGE_WIDTH - 3, IMAGE_HEIGHT - 2, slice(38, 40), slice(57, 60), id="bottom-right-overflow"),
    ],
)
def test_mask_is_clipped_to_the_image(x, y, rows, cols):
    data = np.ones((6, 8), dtype=bool)

    full_mask = decoded_full_mask(context_mask(data, x=x, y=y))

    expected = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected[rows, cols] = 255
    assert full_mask.tolist() == expected.tolist()


def test_single_pixel_mask_at_bottom_right_lands_on_the_final_pixel_only():
    data = np.ones((1, 1), dtype=bool)

    full_mask = decoded_full_mask(
        context_mask(data, x=IMAGE_WIDTH - 1, y=IMAGE_HEIGHT - 1)
    )

    assert np.count_nonzero(full_mask) == 1
    assert full_mask[IMAGE_HEIGHT - 1, IMAGE_WIDTH - 1] == 255


def test_single_pixel_mask_at_image_origin_lands_on_the_first_pixel():
    data = np.ones((1, 1), dtype=bool)

    full_mask = decoded_full_mask(context_mask(data, x=0, y=0))

    assert np.count_nonzero(full_mask) == 1
    assert full_mask[0, 0] == 255


@pytest.mark.parametrize(
    "x, y",
    [
        pytest.param(IMAGE_WIDTH, 0, id="right-of-image"),
        pytest.param(-1, 0, id="left-of-image"),
        pytest.param(0, -1, id="above-image"),
        pytest.param(0, IMAGE_HEIGHT, id="below-image"),
    ],
)
def test_single_pixel_mask_outside_the_image_is_rejected(x, y):
    with pytest.raises(init_mask.InitMaskError, match="no pixels inside"):
        init_mask.decode_context_mask(
            context_mask(np.ones((1, 1), dtype=bool), x=x, y=y),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )


def test_oversize_mask_at_negative_origin_uses_source_offsets():
    data = (np.arange(50)[:, None] * 80 + np.arange(80)[None, :]) % 3 == 0

    full_mask = decoded_full_mask(context_mask(data, x=-5, y=-7))

    expected = data[7 : 7 + IMAGE_HEIGHT, 5 : 5 + IMAGE_WIDTH].astype(np.uint8) * 255
    assert full_mask.tolist() == expected.tolist()


def test_oversize_mask_past_bottom_right_keeps_in_bounds_sub_block():
    data = (np.arange(50)[:, None] * 80 + np.arange(80)[None, :]) % 3 == 0

    full_mask = decoded_full_mask(
        context_mask(data, x=IMAGE_WIDTH - 4, y=IMAGE_HEIGHT - 6)
    )

    expected = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected[IMAGE_HEIGHT - 6 :, IMAGE_WIDTH - 4 :] = data[:6, :4].astype(np.uint8) * 255
    assert full_mask.tolist() == expected.tolist()


def test_the_decoded_bitmap_is_positioned_in_image_coordinates():
    data = np.ones((4, 5), dtype=bool)

    bitmap = init_mask.decode_context_mask(
        context_mask(data, x=11, y=9), IMAGE_HEIGHT, IMAGE_WIDTH
    )

    assert isinstance(bitmap, sly.Bitmap)
    assert (bitmap.origin.col, bitmap.origin.row) == (11, 9)
    assert bitmap.data.tolist() == data.tolist()


def test_a_mask_completely_outside_of_the_image_is_rejected():
    with pytest.raises(init_mask.InitMaskError, match="no pixels inside"):
        init_mask.decode_context_mask(
            context_mask(np.ones((4, 4), dtype=bool), x=IMAGE_WIDTH + 5, y=2),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )


def test_an_all_zero_mask_is_rejected():
    with pytest.raises(init_mask.InitMaskError, match="no pixels inside"):
        init_mask.decode_context_mask(
            context_mask(np.zeros((4, 4), dtype=bool), x=1, y=1),
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )


@pytest.mark.parametrize(
    "mask_json, message",
    [
        pytest.param("not-an-object", "must be an object", id="not-an-object"),
        pytest.param(
            {"data": encode_mask(np.ones((2, 2), dtype=bool))}, "origin", id="origin-missing"
        ),
        pytest.param(
            {"origin": [1], "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "\\[x, y\\] pixel pair",
            id="origin-too-short",
        ),
        pytest.param(
            {"origin": {"x": 1, "y": 2}, "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "\\[x, y\\] pixel pair",
            id="origin-object",
        ),
        pytest.param(
            {"origin": ["1", 2], "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "must be a number",
            id="origin-string",
        ),
        pytest.param(
            {"origin": [1.5, 2], "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "whole pixel coordinate",
            id="origin-fractional",
        ),
        pytest.param(
            {"origin": [float("nan"), 2], "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "finite pixel coordinate",
            id="origin-nan",
        ),
        pytest.param(
            {"origin": [1, float("inf")], "data": encode_mask(np.ones((2, 2), dtype=bool))},
            "finite pixel coordinate",
            id="origin-infinite",
        ),
        pytest.param({"origin": [1, 2]}, "non-empty base64", id="data-missing"),
        pytest.param({"origin": [1, 2], "data": "   "}, "non-empty base64", id="data-blank"),
        pytest.param({"origin": [1, 2], "data": 42}, "non-empty base64", id="data-not-a-string"),
        pytest.param(
            {"origin": [1, 2], "data": "not-a-valid-bitmap-payload"},
            "not a valid encoded bitmap",
            id="data-malformed-base64",
        ),
    ],
)
def test_malformed_mask_input_is_an_explicit_error(mask_json, message):
    with pytest.raises(init_mask.InitMaskError, match=message):
        init_mask.decode_context_mask(mask_json, IMAGE_HEIGHT, IMAGE_WIDTH)


def test_integer_valued_float_origins_are_accepted():
    data = np.ones((2, 3), dtype=bool)

    full_mask = decoded_full_mask({"origin": [4.0, 6.0], "data": encode_mask(data)})

    expected = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected[6:8, 4:7] = 255
    assert full_mask.tolist() == expected.tolist()


def test_bitmap_to_mask_clips_instead_of_raising():
    bitmap = sly.Bitmap(
        np.ones((6, 6), dtype=bool), origin=sly.PointLocation(row=IMAGE_HEIGHT - 2, col=-3)
    )

    full_mask = init_mask.bitmap_to_mask(bitmap, IMAGE_HEIGHT, IMAGE_WIDTH)

    expected = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    expected[IMAGE_HEIGHT - 2 :, 0:3] = 255
    assert full_mask.tolist() == expected.tolist()
    assert set(np.unique(full_mask)) <= {0, 255}


# --------------------------------------------------------------------------- #
# deprecated legacy figure-id download
# --------------------------------------------------------------------------- #
class LegacyApi:
    def __init__(self, objects):
        self.calls = []
        self.objects = objects
        self.annotation = self

    def download_json(self, image_id):
        self.calls.append(image_id)
        return {"objects": list(self.objects)}


def test_legacy_figure_id_download_still_returns_the_stored_bitmap():
    data = np.ones((5, 4), dtype=bool)
    api = LegacyApi([bitmap_label(data, row=6, col=8, figure_id=FIGURE_ID)])

    bitmap = init_mask.download_init_mask(api, FIGURE_ID, IMAGE_ID)

    assert api.calls == [IMAGE_ID]
    assert (bitmap.origin.col, bitmap.origin.row) == (8, 6)
    assert init_mask.bitmap_to_mask(bitmap, IMAGE_HEIGHT, IMAGE_WIDTH).tolist() == (
        init_mask.place_on_image(data, (8, 6), IMAGE_HEIGHT, IMAGE_WIDTH) * 255
    ).astype(np.uint8).tolist()


def test_legacy_download_logs_a_deprecation_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        init_mask.logger, "warn", lambda message, *args, **kwargs: warnings.append(message)
    )
    api = LegacyApi([bitmap_label(np.ones((2, 2), dtype=bool), row=1, col=1)])

    init_mask.download_init_mask(api, FIGURE_ID, IMAGE_ID)

    assert any("deprecated" in message for message in warnings)


def test_legacy_download_without_a_figure_id_is_an_explicit_error():
    api = LegacyApi([])

    with pytest.raises(init_mask.InitMaskError, match="without a usable 'figure_id'"):
        init_mask.download_init_mask(api, None, IMAGE_ID)

    assert api.calls == []


def test_legacy_download_of_a_missing_figure_is_an_explicit_error():
    api = LegacyApi([bitmap_label(np.ones((2, 2), dtype=bool), row=1, col=1, figure_id=1)])

    with pytest.raises(init_mask.InitMaskError, match="Could not download initial figure"):
        init_mask.download_init_mask(api, FIGURE_ID, IMAGE_ID)
