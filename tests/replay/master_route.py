"""Loads the baseline ``/smart_segmentation`` route, verbatim.

``src/main.py`` cannot be imported offline: at import time it instantiates the
Supervisely API client and builds the SAM 2 model. To reproduce the baseline
behavior without a GPU or an instance, the route function is taken from the
exact baseline commit, lifted out of ``SegmentAnything2.serve()`` (its only
free variable is ``self``) and executed against the same offline stand-ins the
head scenarios use. The route body itself is never retyped or modified.

The extracted source is committed next to this module as
``baseline_route_14266f6.py`` so the reproduction also replays in a checkout
that cannot read the baseline commit (for example a shallow clone). Whenever
the commit *is* readable, the fixture is compared against it byte for byte, so
the committed copy can never drift from the history it claims to be.
"""

import ast
import hashlib
import os
import subprocess
import textwrap
from pathlib import Path

#: Baseline commit reproduced by ``baseline_direct_mask.py``: this branch's
#: merge base with ``master``, an ancestor of the current head, so the
#: reproduction reads real history. Its route body is unchanged at current
#: ``master`` head ``b1116682802c11a66c408837569937b68e7979d6``; the loader
#: re-checks that at runtime against ``VERIFY_BASE_SHA``.
BASELINE_SHA = "14266f61aa5e3c118889f3802dfa7e47255c257f"

ROUTE_PATH = "/smart_segmentation"

#: Verbatim route source of ``BASELINE_SHA`` and its digest.
FIXTURE_PATH = Path(__file__).with_name("baseline_route_14266f6.py")
FIXTURE_SHA256 = "461918e30d51d261e534a1d8422d915daeb3c2b0d344e9118e55a9291e120366"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _decorated_route(module):
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


def _route_source_at(sha):
    """Returns the route source of ``sha:src/main.py``, or ``None`` and a reason."""
    show = _git("show", f"{sha}:src/main.py")
    if show.returncode != 0:
        return None, (show.stderr or "commit unavailable").strip()
    node = _decorated_route(ast.parse(show.stdout))
    source = textwrap.dedent(ast.get_source_segment(show.stdout, node, padded=True))
    return source.rstrip() + "\n", None


def baseline_route_source(sha=BASELINE_SHA):
    """Returns ``(source, verified_against_git)`` for the baseline route."""
    fixture = FIXTURE_PATH.read_text()
    digest = hashlib.sha256(fixture.encode()).hexdigest()
    assert digest == FIXTURE_SHA256, f"baseline route fixture was modified: {digest}"

    verify_base = os.environ.get("VERIFY_BASE_SHA")
    if verify_base and not sha.startswith(verify_base):
        # The reproduced commit is this branch's merge base. Report whether the
        # runtime's base commit still carries the very same route, so the
        # reproduction can never quietly describe an outdated baseline.
        base_source, reason = _route_source_at(verify_base)
        if base_source is None:
            state = f"not readable here ({reason})"
        elif base_source == fixture:
            state = "byte-identical to the reproduced route"
        else:
            state = "DIFFERENT from the reproduced route"
        print(
            f"note: runtime base {verify_base} {ROUTE_PATH} route is {state}; "
            f"reproducing pinned {sha}"
        )

    ancestry = _git("merge-base", "--is-ancestor", sha, "HEAD")
    extracted, reason = _route_source_at(sha)
    if ancestry.returncode != 0 or extracted is None:
        reason = ancestry.stderr.strip() or reason or "commit unavailable"
        print(f"note: baseline commit {sha} unreadable here ({reason}); using the committed fixture")
        return fixture, False

    assert extracted == fixture, (
        f"the committed baseline route fixture does not match {sha}:src/main.py"
    )
    return fixture, True


def load_baseline_route(sha=BASELINE_SHA):
    """Returns the baseline route as ``route(self, response, request)``."""
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

    source, verified = baseline_route_source(sha)
    module = ast.parse(source)
    route = module.body[0]
    assert isinstance(route, ast.FunctionDef) and route.name == "smart_segmentation", source[:80]
    route.decorator_list = []
    route.args.args.insert(0, ast.arg(arg="self"))
    ast.fix_missing_locations(module)
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
    return namespace["smart_segmentation"], verified
