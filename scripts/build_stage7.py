"""Redirect legacy command to comprehensive Stage 7 v2."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_stage7_full.py")),run_name="__main__")
