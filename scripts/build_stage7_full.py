from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_vector import build

if __name__=='__main__':
    meta=build()
    print(f"PASS: source-locked FAISS | {meta['count']} reviewed documents | {meta['dimension']}D")
    print(f"Manifest SHA: {meta['fingerprint']}")
    print('SUCCESS: Stage 7 full reviewed index built; no LLM quota consumed.')
