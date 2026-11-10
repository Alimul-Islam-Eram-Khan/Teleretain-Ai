"""Non-destructively archive obsolete 18-offer Stage 7 tests after installing v2."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    target=ROOT/'reports'/'legacy_stage7_v1_tests'
    for filename in ('test_stage7_catalog.py','test_stage7_rag.py'):
        old=ROOT/'tests'/filename
        if old.exists():
            target.mkdir(parents=True,exist_ok=True)
            destination=target/(filename+".bak")
            if destination.exists():
                print('SKIP: backup exists',destination);continue
            shutil.move(str(old),str(destination))
            print('ARCHIVED:',old,'->',destination)
    print('Only obsolete Stage 7 v1 tests are archived; Stage 1-6 tests stay unchanged.')
