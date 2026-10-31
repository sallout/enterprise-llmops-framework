"""Isolated official garak detector. JSON stdin/stdout; no LLM or endpoint calls."""
import contextlib
import json
import sys
from importlib.metadata import version
from types import SimpleNamespace


def main():
    # Third-party progress/config messages must not corrupt the protocol.
    with contextlib.redirect_stdout(sys.stderr):
        from garak.detectors.leakreplay import StartsWith
        from garak.attempt import Message
        request=json.load(sys.stdin)
        result={'garak_version':version('garak')}
        if request.get('operation')=='detect':
            attempt=SimpleNamespace(notes={'triggers':[request['expected']]},outputs=[Message(text=request['output'])])
            result['scores']=StartsWith().detect(attempt)
    json.dump(result,sys.stdout)


if __name__=='__main__':
    main()
