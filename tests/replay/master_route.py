"""Loads the baseline ``/smart_segmentation`` route from git, verbatim.

``src/main.py`` cannot be imported offline: at import time it instantiates the
Supervisely API client and builds the SAM 2 model. To reproduce the baseline
behavior without a GPU or an instance, the route function is read from the
exact baseline commit, lifted out of ``SegmentAnything2.serve()`` (its only
free variable is ``self``) and executed against the same offline stand-ins the
head scenarios use. The route body itself is never retyped or modified.
"""

import ast
import subprocess

#: Baseline commit reproduced by ``baseline_direct_mask.py`` (``master`` head).
BASELINE_SHA = "ace8b6813f0caa877f39c086137d8a0969748e22"

ROUTE_PATH = "/smart_segmentation"


def baseline_main_source(sha=BASELINE_SHA):
    return subprocess.check_output(["git", "show", f"{sha}:src/main.py"], text=True)


def _find_route(module):
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef) or node.name != "smart_segmentation":
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and decorator.args
                and getattr(decorator.args[0], "value", None) == ROUTE_PATH
            ):
                return node
    raise AssertionError(f"No {ROUTE_PATH} route found in the baseline source")


def load_baseline_route(sha=BASELINE_SHA):
    """Returns the baseline route as ``route(self, response, request)``."""
    import os
    import time

    import supervisely as sly
    from fastapi import Request, Response, status
    from supervisely._utils import rand_str
    from supervisely.app.content import get_data_dir
    from supervisely.imaging import image as sly_image
    from supervisely.io.fs import silent_remove
    from supervisely.nn.inference.interactive_segmentation import functional
    from supervisely.sly_logger import logger

    from src.smart_segmentation import get_plane_name

    route = _find_route(ast.parse(baseline_main_source(sha)))
    route.decorator_list = []
    route.args.args.insert(0, ast.arg(arg="self"))
    module = ast.Module(body=[route], type_ignores=[])
    ast.fix_missing_locations(ast.increment_lineno(module, 0))
    namespace = {
        "os": os,
        "time": time,
        "sly": sly,
        "sly_image": sly_image,
        "status": status,
        "Request": Request,
        "Response": Response,
        "functional": functional,
        "get_data_dir": get_data_dir,
        "get_plane_name": get_plane_name,
        "logger": logger,
        "rand_str": rand_str,
        "silent_remove": silent_remove,
    }
    exec(compile(module, f"<master {sha[:7]}:src/main.py>", "exec"), namespace)
    return namespace["smart_segmentation"]
