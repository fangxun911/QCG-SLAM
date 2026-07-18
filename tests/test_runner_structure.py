"""Structural tests for the QCG-SLAM runner module."""

import ast
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "qcg_slam" / "runner.py"


def _runner_ast():
    return ast.parse(RUNNER_PATH.read_text())


def _runner_class(module):
    return next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "RGBDSLAMRunner")


def test_runner_exposes_class_with_run_method():
    module = _runner_ast()
    runner_class = _runner_class(module)

    method_names = {
        node.name
        for node in runner_class.body if isinstance(node, ast.FunctionDef)
    }

    assert {"__init__", "run"}.issubset(method_names)


def test_runner_run_delegates_to_pipeline_function():
    module = _runner_ast()
    runner_class = _runner_class(module)
    run_method = next(
        node for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run")

    called_methods = [
        node.value.func.attr for node in ast.walk(run_method)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "self"
    ]

    assert called_methods == [
        "prepare",
        "load_datasets",
        "initialize_state",
        "run_frame_loop",
        "run_global_optimization",
        "finalize",
    ]


def test_runner_exposes_pipeline_stage_methods():
    module = _runner_ast()
    runner_class = _runner_class(module)
    method_names = {
        node.name
        for node in runner_class.body if isinstance(node, ast.FunctionDef)
    }

    assert {
        "prepare",
        "load_datasets",
        "initialize_state",
        "run_frame_loop",
        "run_global_optimization",
        "finalize",
    }.issubset(method_names)


def test_rgbd_slam_delegates_to_runner_class():
    module = _runner_ast()
    rgbd_slam_fn = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "rgbd_slam")

    assert len(rgbd_slam_fn.body) == 1
    return_stmt = rgbd_slam_fn.body[0]
    assert isinstance(return_stmt, ast.Return)
    assert isinstance(return_stmt.value, ast.Call)
    assert isinstance(return_stmt.value.func, ast.Attribute)
    assert return_stmt.value.func.attr == "run"
    assert isinstance(return_stmt.value.func.value, ast.Call)
    assert isinstance(return_stmt.value.func.value.func, ast.Name)
    assert return_stmt.value.func.value.func.id == "RGBDSLAMRunner"


def test_legacy_pipeline_function_delegates_to_runner_class():
    module = _runner_ast()
    pipeline_fn = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_rgbd_slam_pipeline")

    return_stmt = next(
        node for node in pipeline_fn.body if isinstance(node, ast.Return))
    assert isinstance(return_stmt.value, ast.Call)
    assert isinstance(return_stmt.value.func, ast.Attribute)
    assert return_stmt.value.func.attr == "run"
    assert isinstance(return_stmt.value.func.value, ast.Call)
    assert isinstance(return_stmt.value.func.value.func, ast.Name)
    assert return_stmt.value.func.value.func.id == "RGBDSLAMRunner"
