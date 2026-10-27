import logging

from typer.testing import CliRunner

from policy_dissertation.cli import app, run_safely
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.logging import configure_logging

runner = CliRunner()


def test_help_does_not_load_configuration(monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'invalid')
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'status' in result.output


def test_status(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LOG_LEVEL', 'INFO')
    result = runner.invoke(app, ['status'])
    assert result.exit_code == 0
    assert 'Document ingestion is available' in result.output


def test_dotenv_and_environment_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ('MODEL__BASE_URL', 'MODEL__API_KEY', 'MODEL__MODEL_NAME', 'POLICY_DIR', 'LOG_LEVEL'):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / '.env').write_text(
        'MODEL__BASE_URL=https://example.invalid/v1\n'
        'MODEL__API_KEY=test-secret\nMODEL__MODEL_NAME=test-model\n'
    )
    monkeypatch.setenv('MODEL__MODEL_NAME', 'environment-model')
    settings = Settings()
    assert settings.model.base_url == 'https://example.invalid/v1'
    assert settings.model.model_name == 'environment-model'
    assert settings.model.api_key.get_secret_value() == 'test-secret'
    assert 'test-secret' not in repr(settings)
    assert 'test-secret' not in settings.model_dump_json()
    assert str(settings.policy_dir) == 'dumy_policies'


def test_invalid_config_does_not_disclose_input(monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'private-invalid-value')
    result = runner.invoke(app, ['status'])
    assert result.exit_code == 2
    assert 'invalid configuration' in result.output
    assert 'private-invalid-value' not in result.output


def test_error_boundary():
    import typer
    import pytest

    def expected():
        raise DissertationError('A safe failure.')

    def unexpected():
        raise RuntimeError('private-details')

    for action in (expected, unexpected):
        with pytest.raises(typer.Exit) as error:
            run_safely(action)
        assert error.value.exit_code == 1


def test_logging_is_idempotent_and_uses_stderr(capsys):
    configure_logging('INFO')
    configure_logging('WARNING')
    logger = logging.getLogger('policy_dissertation')
    logger.info('hidden')
    logger.warning('visible')
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'hidden' not in captured.err
    assert captured.err.count('visible') == 1
    logger.handlers.clear()
