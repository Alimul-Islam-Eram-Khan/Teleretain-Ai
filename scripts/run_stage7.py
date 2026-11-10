"""Default Stage 7 command now routes to fresh official-source retrieval."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('run_stage7_live.py')),run_name='__main__')
