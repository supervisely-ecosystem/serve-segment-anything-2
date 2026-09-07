"""Metadata regressions for the root serving app.

The Smart Tool change must not advertise new capabilities: ``allowed_shapes``
and the session tags used by tracking stay exactly as they are.
"""

import ast
import json
import os
import re

import pytest

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


def test_dockerfile_sdk_pin_matches_the_image_label():
    with open(os.path.join(REPO_ROOT, "docker", "Dockerfile"), "r") as file:
        dockerfile = file.read()

    installed = re.search(r"supervisely==([0-9.]+)", dockerfile)
    labeled = re.search(r'python_sdk_version="([0-9.]+)"', dockerfile)

    assert installed is not None and labeled is not None
    # the adapter must keep working with the SDK bundled into this image
    assert installed.group(1) == labeled.group(1)


def test_smart_segmentation_route_calls_the_extracted_handler():
    """``src/main.py`` must delegate the route to the tested handler."""
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
    called = {
        node.func.id
        for node in ast.walk(routes[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called == {"smart_segmentation_handler"}
