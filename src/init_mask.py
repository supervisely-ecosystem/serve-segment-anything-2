"""The init mask of a Smart Tool request.

The canonical implementation is ``get_init_mask_from_context`` in the Supervisely
SDK (``supervisely/nn/inference/interactive_segmentation/functional.py``). This
app cannot import it: ``docker/Dockerfile`` installs
``supervisely[video-av]==6.74.36`` and ``config.json`` pins
``supervisely/segment-anything-2:1.0.21``, both of which predate that helper, so
it is mirrored here -- same field names, same origin convention, same decoding,
same error behaviour -- on top of APIs the pinned SDK already has. Keep the two
in step; when the pin moves past the release that carries the helper, drop this
module and import the SDK one.

The app needs the mask of the figure being refined as ``settings["init_mask"]``.
Historically the request only named the figure (``figure_id`` plus
``init_figure``) and the app resolved it by downloading the image annotation and
parsing the label back as a ``sly.Bitmap``, which only ever worked for bitmap
figures. The platform now sends the mask itself, so a polygon, multipolygon or
AnyShape figure can be handed back to the model as well.
"""

from typing import Optional

import supervisely as sly


class InitMaskDecodeError(ValueError):
    """Request context carries an init ``mask`` that cannot be decoded."""


def get_init_mask_from_context(context: dict) -> Optional[sly.Bitmap]:
    """Build the init mask bitmap from the ``mask`` field of the request context.

    The field is optional and has the form
    ``{"data": <base64 string>, "origin": {"x": <int>, "y": <int>}}``, where
    ``data`` is the same encoding ``sly.Bitmap`` uses on the wire (base64 of a
    zlib-compressed PNG, non-zero pixels are foreground) and ``origin`` is the
    top-left corner of the mask in full image coordinates. When it is present it
    fully replaces the deprecated ``figure_id`` lookup, so no annotation is
    downloaded and figures of any geometry can be sent back.

    :param context: Request context of a smart tool request.
    :type context: dict
    :returns: Bitmap built from the context, or None if there is no ``mask``.
    :rtype: :class:`supervisely.Bitmap` or None
    :raises InitMaskDecodeError: if ``mask`` is present but cannot be decoded.
    """
    mask = context.get("mask")
    if mask is None:
        return None
    try:
        origin = mask["origin"]
        data = sly.Bitmap.base64_2_data(mask["data"])
        return sly.Bitmap(
            data=data,
            origin=sly.PointLocation(row=int(origin["y"]), col=int(origin["x"])),
        )
    except Exception as exc:
        raise InitMaskDecodeError(f"Can not decode the init mask from request: {exc}") from exc
