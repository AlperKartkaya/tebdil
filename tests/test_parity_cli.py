import runpy
from pathlib import Path
import sys
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/check_cached_parity.py'


@pytest.mark.parametrize('arguments', [['--tokens', '0'], ['--prefix-lengths', '-1']])
def test_invalid_parity_dimensions_fail_before_loading(monkeypatch, tmp_path, arguments):
    namespace = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--output', str(tmp_path / 'report.json')] + arguments)
    with pytest.raises(SystemExit) as error:
        namespace['main']()
    assert error.value.code == 2
