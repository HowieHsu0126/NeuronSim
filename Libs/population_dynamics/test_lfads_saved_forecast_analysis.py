"""Verify delivered-input staging and archive integrity without forecasts."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_lfads_saved_forecast_analysis as runner


def test_software_receipt_does_not_replace_archive_delivery(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'module.py').write_text('value=1\n')
    row=dict(path='module.py',bytes=(source/'module.py').stat().st_size,sha256=runner.digest(source/'module.py'))
    (source/'delivery.json').write_text(json.dumps(dict(status='PASS',files=[row])))
    work=tmp_path/'work';work.mkdir()
    runner.stage_receipt(source,work,Path('.'),['module.py'])
    assert not (work/'delivery.json').exists()
    assert (work/'software_delivery.json').read_bytes()==(source/'delivery.json').read_bytes()
    output=tmp_path/'output';record=dict(status='PASS')
    runner.archive(work,output,record)
    proof=json.loads((output/'delivery.json').read_text())
    assert 'delivery.json' not in {row['path'] for row in proof['files']}
    assert all(runner.digest(output/row['path'])==row['sha256'] for row in proof['files'])
    (source/'module.py').write_text('changed\n')
    with pytest.raises(ValueError,match='source manifest identity changed'):
        runner.stage_receipt(source,tmp_path/'rejected',Path('.'),['module.py'])
