"""No host execution of repository/generated code. Python runs only in constrained Docker."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

from .schemas import ExecutionResult

# Static preflight restricts the supported execution subset, not a sandbox on its own.
ALLOWED_IMPORTS = {'solution', 'unittest', 'math', 're', 'json', 'datetime', 'collections', 'itertools',
                   'functools', 'typing', 'dataclasses', 'decimal', 'fractions', 'statistics', 'string',
                   'enum', 'hashlib', 'hmac', 'secrets', 'base64', 'bisect', 'heapq', 'copy', 'operator'}
BLOCKED_CALLS = {'exec', 'eval', 'compile', '__import__', 'open', 'input', 'globals', 'locals', 'vars',
                 'getattr', 'setattr', 'delattr', 'breakpoint', 'exit', 'quit'}


def execution_preflight(solution: str, tests: str) -> str | None:
    try:
        trees = [ast.parse(solution), ast.parse(tests)]
    except (SyntaxError, ValueError, RecursionError):
        return 'invalid_python_syntax'
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or '').split('.')[0]]
                if any(name not in ALLOWED_IMPORTS for name in names):
                    return 'unsupported_or_restricted_import'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                return 'restricted_builtin'
            if isinstance(node, ast.Attribute) and node.attr in ('skip', 'skipIf', 'skipUnless', 'expectedFailure'):
                return 'skipped_tests_not_allowed'
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                return 'restricted_introspection'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('skipTest', 'skip', 'skipIf', 'skipUnless', 'expectedFailure'):
                return 'skipped_tests_not_allowed'
    test_tree = trees[1]
    tests_found = [n for n in ast.walk(test_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_')]
    assertions = [n for n in ast.walk(test_tree) if isinstance(n, ast.Assert) or (
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith('assert'))]
    imports_solution = any(isinstance(n, ast.ImportFrom) and n.module == 'solution' or
                           isinstance(n, ast.Import) and any(a.name == 'solution' for a in n.names)
                           for n in ast.walk(test_tree))
    if not tests_found or not assertions or not imports_solution:
        return 'tests_must_import_solution_and_assert_behavior'
    return None


class DockerSandbox:
    def __init__(self, config):
        self.config = config
        self.available = False
        self.reason = 'execution_disabled'
        self.image_id = None
        if config.execution_mode == 'disabled':
            return
        try:
            daemon = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], capture_output=True, timeout=8)
            if daemon.returncode:
                self.reason = 'docker_daemon_unavailable'
                return
            image = subprocess.run(['docker', 'image', 'inspect', config.python_image, '--format', '{{.Id}}'], capture_output=True, text=True, timeout=8)
            if image.returncode:
                self.reason = 'configured_python_image_not_local'
                return
            self.image_id = image.stdout.strip()
            if not re.fullmatch(r'sha256:[a-f0-9]{64}', self.image_id):
                self.reason = 'image_identity_unavailable'
                return
            self.available, self.reason = True, 'restricted_docker_available'
        except (OSError, subprocess.TimeoutExpired):
            self.reason = 'docker_runtime_unavailable'

    def run(self, language: str, solution: str, tests: str) -> ExecutionResult:
        if language != 'python':
            return ExecutionResult(status='unverified', executor='none', reason='language_runtime_not_configured')
        restricted = execution_preflight(solution, tests)
        if restricted:
            return ExecutionResult(status='unverified', executor='none', reason=restricted)
        if not self.available:
            return ExecutionResult(status='unverified', executor='none', reason=self.reason)
        name = 'pip2-'+uuid.uuid4().hex
        nonce = uuid.uuid4().hex
        command = ['python', '-I', '/work/runner.py']
        runner = '''import io,json,sys,unittest
sys.path.insert(0,"/work")
stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromName("test_solution")
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
payload={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"expected_failures":len(result.expectedFailures),"unexpected_successes":len(result.unexpectedSuccesses),"feedback":stream.getvalue()[:LIMIT]}
print("NONCE"+json.dumps(payload),flush=True)
sys.exit(0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped else 1)
'''.replace('NONCE', nonce).replace('LIMIT', str(self.config.max_feedback_bytes))
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='pip2-sandbox-') as directory:
            root = Path(directory)
            root.chmod(0o755)
            for filename, text in [('solution.py', solution), ('test_solution.py', tests), ('runner.py', runner)]:
                (root/filename).write_text(text, encoding='utf-8')
                (root/filename).chmod(0o444)
            create = ['docker', 'create', '--name', name, '--network', 'none', '--read-only',
                      '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534',
                      '--memory', '128m', '--memory-swap', '128m', '--cpus', '1', '--pids-limit', '32',
                      '--ulimit', 'fsize=1048576:1048576', '--ulimit', 'nofile=64:64',
                      '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
                      '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1',
                      '--mount', f'type=bind,src={root},dst=/work,readonly',
                      '--env', 'PYTHONDONTWRITEBYTECODE=1', '--workdir', '/work', self.image_id, *command]
            try:
                created = subprocess.run(create, capture_output=True, timeout=10)
                if created.returncode:
                    return ExecutionResult(status='unverified', executor='docker', reason='sandbox_creation_failed')
                subprocess.run(['docker', 'start', name], capture_output=True, timeout=10, check=True)
                waited = subprocess.run(['docker', 'wait', name], capture_output=True, text=True, timeout=self.config.sandbox_timeout)
                logs = subprocess.run(['docker', 'logs', '--tail', '40', name], capture_output=True, timeout=5)
                output = logs.stdout[-self.config.max_feedback_bytes*2:].decode('utf-8', errors='replace')
                payload = None
                for line in output.splitlines():
                    if line.startswith(nonce):
                        payload = json.loads(line[len(nonce):])
                exit_code = int(waited.stdout.strip())
                if not payload:
                    return ExecutionResult(status='failed', executor='docker', reason='runner_result_missing',
                                           exit_code=exit_code, command=command, image_id=self.image_id,
                                           feedback=output[-self.config.max_feedback_bytes:])
                passed = (exit_code == 0 and payload['tests_run'] > 0 and not any(payload[k] for k in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes')))
                return ExecutionResult(status='verified' if passed else 'failed', executor='docker',
                                       reason='tests_passed' if passed else 'tests_failed', command=command,
                                       exit_code=exit_code, image_id=self.image_id,
                                       duration_seconds=round(time.monotonic()-start, 3), **payload)
            except subprocess.TimeoutExpired:
                return ExecutionResult(status='failed', executor='docker', reason='sandbox_timeout', command=command,
                                       image_id=self.image_id, feedback='Container deadline exceeded; execution terminated.')
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
                return ExecutionResult(status='unverified', executor='docker', reason='sandbox_runtime_error', command=command)
            finally:
                try:
                    subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
