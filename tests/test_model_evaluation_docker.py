"""Opt-in real isolated execution, never executes candidate code on the host.

RUN_MODEL_EVALUATION_DOCKER_TESTS=1 uv run pytest tests/test_model_evaluation_docker.py -q
Requires a Docker daemon and local python:3.12-slim image.
"""
import os
import pytest

pytestmark=pytest.mark.skipif(os.getenv('RUN_MODEL_EVALUATION_DOCKER_TESTS')!='1',reason='Opt-in real Docker evaluation isolation')


def test_real_pass_failure_timeout_and_premature_exit():
    from policy_dissertation.pipelines.model_evaluation.execution import DockerExecutor
    from policy_dissertation.pipelines.model_evaluation.schemas import EvaluationConfig
    executor=DockerExecutor(EvaluationConfig(timeout_seconds=2));executor.preflight()
    assert executor.run('def add(a,b): return a+b','assert add(2,3)==5')['status']=='passed'
    assert executor.run('def add(a,b): return a-b','assert add(2,3)==5')['status']=='failed'
    assert executor.run('import sys\nsys.exit(0)','assert False')['status']=='failed'
    assert executor.run('while True: pass','assert True')['reason']=='timeout'
