"""New default preflight: validate live-source dependencies and guards."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('preflight_stage7_live.py')),run_name='__main__')
