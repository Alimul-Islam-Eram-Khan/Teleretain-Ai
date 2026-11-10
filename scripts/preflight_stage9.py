"""Read-only Stage 9 preflight. Does not call Teletalk or any LLM."""
from pathlib import Path
import ast
import importlib
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.dashboard_stage9 import checked_track, DATA_POLICY


def main():
    script=root/'app'/'streamlit_app.py'
    ast.parse(script.read_text(encoding='utf-8'))
    print('PASS: Stage 9 dashboard Python syntax')
    for name in ('streamlit','plotly','pandas','pydantic'):
        mod=importlib.import_module(name)
        print(f'PASS: {name} import ({getattr(mod,"__version__","available")})')
    for track in ('iranian','ibm'):
        checked_track(track)
        for path in (root/'models'/f'{track}_best_model.pkl',
                     root/'reports'/'stage3'/f'{track}_results.json',
                     root/'data'/'processed'/track/'val.csv'):
            if not path.is_file():raise FileNotFoundError(f'Stage 9 prerequisite missing: {path}')
        print(f'PASS: {track} trusted model, report and public validation split present')
    for p in ('src/stage8_chat.py','src/retention_recommender.py','src/explain.py','src/routing.py'):
        if not (root/p).is_file():raise FileNotFoundError(f'Prerequisite missing: {p}')
    print('SUCCESS: Stage 9 dashboard preflight PASS (no live website/API calls).')

if __name__=='__main__':main()
