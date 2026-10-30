"""Bounded Python execution in a local Docker daemon; no host-code fallback."""
import ast
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from policy_dissertation.errors import DissertationError


def extract_code(text):
    """One unambiguous code fence or plain code; never repair/truncate a solution."""
    fences = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    if len(fences) == 1:
        return fences[0].strip() + '\n'
    if len(fences) > 1 or '```' in text:
        return None
    return text.strip() + '\n'


class DockerExecutor:
    def __init__(self, config):
        self.config = config
        self.image_id = None

    def preflight(self):
        try:
            result = subprocess.run(['docker', 'image', 'inspect', self.config.python_image, '--format', '{{.Id}}'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode or not re.fullmatch(r'sha256:[0-9a-f]{64}', result.stdout.strip()):
                raise DissertationError('Docker daemon/local Python image unavailable. Prepare the configured image first; generated code is never executed on the host. RunPod containers may not provide a Docker daemon. Use --executor disabled for unscored generation only, or an evaluation host with Docker.')
            self.image_id = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            raise DissertationError('Docker runtime unavailable. Full code-test evaluation requires a Docker host; --executor disabled produces no pass@1 score.') from None
        return {'executor': 'docker', 'image_id': self.image_id, 'timeout_seconds': self.config.timeout_seconds}

    def run(self, code, tests):
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            return {'status': 'failed', 'reason': 'syntax_error'}
        name = 'llmops-eval-' + uuid.uuid4().hex
        marker = 'EVAL_OK_' + uuid.uuid4().hex
        # Tests execute after candidate definitions and must reach the final marker.
        # The container receives only this task, with no credentials or GPU devices.
        script = ('namespace={}\nexec(compile(' + repr(code) + ",'<candidate>','exec'),namespace)\n"
                  + 'exec(compile(' + repr(tests) + ",'<tests>','exec'),namespace)\nprint(" + repr(marker) + ',flush=True)\n')
        try:
            with tempfile.TemporaryDirectory(prefix='llmops-eval-') as directory:
                folder = Path(directory); folder.chmod(0o755)
                path = folder / 'task.py'; path.write_text(script); path.chmod(0o444)
                command = ['docker', 'create', '--name', name, '--network', 'none', '--read-only',
                    '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534',
                    '--memory', '512m', '--memory-swap', '512m', '--cpus', '1', '--pids-limit', '32',
                    '--ulimit', 'fsize=1048576:1048576', '--ulimit', 'nofile=64:64',
                    '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1',
                    '--log-opt', 'compress=false',
                    '--mount', f'type=bind,src={folder},dst=/evaluation,readonly',
                    self.image_id, 'python', '-I', '-B', '/evaluation/task.py']
                created = subprocess.run(command, capture_output=True, timeout=15)
                if created.returncode:
                    return {'status': 'unverified', 'reason': 'container_create_failed'}
                started = subprocess.run(['docker', 'start', name], capture_output=True, timeout=10)
                if started.returncode:
                    return {'status': 'unverified', 'reason': 'container_start_failed'}
                waited = subprocess.run(['docker', 'wait', name], capture_output=True, text=True,
                                        timeout=self.config.timeout_seconds)
                if waited.returncode:
                    return {'status': 'unverified', 'reason': 'container_wait_failed'}
                logs = subprocess.run(['docker', 'logs', '--tail', '30', name], capture_output=True, text=True, timeout=10)
                passed = waited.stdout.strip() == '0' and marker in logs.stdout.splitlines()
                return {'status': 'passed' if passed else 'failed', 'reason': 'tests_passed' if passed else 'tests_failed',
                        'exit_code': waited.stdout.strip(), 'image_id': self.image_id}
        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'reason': 'timeout'}
        except OSError:
            return {'status': 'unverified', 'reason': 'docker_runtime_error'}
        finally:
            try:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
