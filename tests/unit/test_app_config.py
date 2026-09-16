"""Metadata regressions for the root serving app.

The Smart Tool change must not advertise new capabilities: ``allowed_shapes``
and the session tags used by tracking stay exactly as they are.
"""

import ast
import json
import os
from inspect import signature

import pytest

from src.smart_segmentation import smart_segmentation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def config():
    with open(os.path.join(REPO_ROOT, "config.json"), "r") as file:
        return json.load(file)


def test_allowed_shapes_are_not_extended(config):
    assert config["allowed_shapes"] == ["bitmap"]


def test_session_tags_are_unchanged(config):
    assert config["session_tags"] == [
        "sly_smart_annotation",
        "deployed_nn_object_segmentation",
        "sly_video_tracking",
        "scalable",
    ]


def test_app_still_declares_its_own_docker_image_and_entrypoint(config):
    assert config["docker_image"].startswith("supervisely/segment-anything-2:")
    assert "src.main:m.app" in config["entrypoint"]


def test_smart_segmentation_route_calls_the_extracted_handler():
    """``src/main.py`` must delegate the route to the tested handler.

    ``src/main.py`` cannot be imported offline (it instantiates the API client
    and the SAM 2 model at import time), so the production route wiring is
    asserted structurally: the tested handler is what the route calls.
    """
    with open(os.path.join(REPO_ROOT, "src", "main.py"), "r") as file:
        module = ast.parse(file.read())

    imported = {
        alias.asname or alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module == "src.smart_segmentation"
        for alias in node.names
    }
    assert "smart_segmentation_handler" in imported

    routes = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "smart_segmentation"
        and any(
            isinstance(decorator, ast.Call)
            and decorator.args
            and getattr(decorator.args[0], "value", None) == "/smart_segmentation"
            for decorator in node.decorator_list
        )
    ]
    assert len(routes) == 1
    assert [argument.arg for argument in routes[0].args.args] == ["response", "request"]

    calls = [
        node
        for node in ast.walk(routes[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert len(calls) == 1
    call = calls[0]
    assert call.func.id == "smart_segmentation_handler"
    # the served model, the response and the request reach the handler in the
    # order its signature expects; a swapped argument cannot be caught by the
    # handler tests themselves
    assert [getattr(argument, "id", None) for argument in call.args] == [
        "self",
        "response",
        "request",
    ]
    assert not call.keywords
    assert list(signature(smart_segmentation).parameters) == ["model", "response", "request"]
