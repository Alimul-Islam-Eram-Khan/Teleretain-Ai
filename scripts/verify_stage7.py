"""New default verifier: use --check-live to test Teletalk accessibility."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('verify_stage7_live.py')),run_name='__main__')
