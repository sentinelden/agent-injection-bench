"""The command line, as the README documents it."""

import subprocess
import sys


def test_the_documented_http_target_resolves_from_the_cli():
    # A fresh interpreter on purpose: a test that imports the adapter registers
    # it for the whole process, which is how this stayed broken with tests green.
    code = "from aib.cli import _make_target; print(_make_target('openai-compat:model=m').name)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "openai-compat:m"
