"""Explicit, backed-up environment editing without logging or returning secret values."""
import os
from pathlib import Path
import shutil
import tempfile
from dotenv import set_key
from pydantic import HttpUrl, TypeAdapter, ValidationError
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError

ALLOWED_KEYS={'MODEL__BASE_URL','MODEL__API_KEY','MODEL__MODEL_NAME','HF_TOKEN','LOG_LEVEL',
              'UI__TITLE','UI__AUTHOR','UI__LINKEDIN_URL',
              'PIPELINE1__RESPONSE_MAX_TOKENS','PIPELINE2__RESPONSE_MAX_TOKENS',
              'PIPELINE1__REQUEST_TIMEOUT','PIPELINE2__REQUEST_TIMEOUT',
              'PIPELINE1__MAX_ATTEMPTS','PIPELINE2__MAX_ATTEMPTS','PIPELINE3__MAX_ATTEMPTS',
              'PIPELINE3__REQUEST_TIMEOUT','PIPELINE3__SCAN_MULTIPLIER'}


class FileSettings(Settings):
    @classmethod
    def settings_customise_sources(cls,settings_cls,init_settings,env_settings,dotenv_settings,file_secret_settings):
        # Validate the saved file itself, without process overrides hiding an invalid edit.
        return init_settings,dotenv_settings


def save_environment(path,changes):
    """Preserve unspecified lines; create a 0600 backup before atomic replacement.

    Called only after the operator explicitly confirms saving in the interactive UI.
    The function returns paths/key names, never credentials. It makes no network calls.
    """
    path=Path(path).absolute()
    if path.is_symlink():raise DissertationError('Refusing to edit a symlinked environment file.')
    if not changes or any(k not in ALLOWED_KEYS for k in changes):raise DissertationError('No supported environment changes supplied.')
    for key,value in changes.items():
        if not isinstance(value,str) or not value or any(ord(c)<32 or ord(c)==127 for c in value) or '${' in value:
            raise DissertationError('Environment values must be nonempty single-line literals.')
        if key in ('MODEL__BASE_URL','UI__LINKEDIN_URL'):
            try:TypeAdapter(HttpUrl).validate_python(value)
            except ValidationError:raise DissertationError('Use a valid HTTP(S) URL.') from None
        if key=='LOG_LEVEL' and value not in ('DEBUG','INFO','WARNING','ERROR','CRITICAL'):
            raise DissertationError('Choose a supported log level.')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None;backup=None
    try:
        fd,name=tempfile.mkstemp(prefix='.env.llmops-',dir=path.parent)
        os.close(fd);temporary=Path(name)
        if path.exists():shutil.copyfile(path,temporary)
        for key,value in changes.items():set_key(temporary,key,value,quote_mode='always')
        FileSettings(_env_file=temporary)
        os.chmod(temporary,0o600)
        if path.exists():
            fd,name=tempfile.mkstemp(prefix='.env.backup-',dir=path.parent)
            os.close(fd);backup=Path(name)
            shutil.copyfile(path,backup);os.chmod(backup,0o600)
        temporary.replace(path);temporary=None
    except ValidationError:
        raise DissertationError('Environment configuration is invalid; the original file was preserved.') from None
    except OSError:
        raise DissertationError('Could not save environment configuration; check file permissions and available disk space.') from None
    finally:
        if temporary:temporary.unlink(missing_ok=True)
    return {'path':str(path),'backup':str(backup) if backup else None,'updated_keys':sorted(changes),
            'process_overrides_present':any(k in os.environ for k in changes)}


def environment_status():
    """Presence/configuration checks only; never imply endpoint connectivity was tested."""
    try:
        settings=Settings()
        return {'configuration_valid':True,'env_file_exists':Path('.env').is_file(),
                'llm_endpoint':bool(settings.model.base_url),'llm_api_key':bool(settings.model.api_key),
                'llm_model':bool(settings.model.model_name),'hugging_face_token':bool(settings.hf_token)}
    except (ValidationError,ValueError):
        return {'configuration_valid':False,'env_file_exists':Path('.env').is_file()}
