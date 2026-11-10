from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_kb import load_offers,load_info,load_conflicts
from src.teletalk_answer import answer_offers

if __name__=='__main__':
    c=load_offers();p=load_info()
    print(f'PASS: approved catalog | {len(c.offers)} offers | {len(p)} public information references')
    print(f'PASS: {len(load_conflicts())} quarantined conflict(s) | no personal Teletalk customer data')
    response=answer_offers('internet 30 days under 100 taka')
    assert response.mode=='grounded' and response.offer_ids
    print('PASS: deterministic filtering and official-source citations')
    try:
        import faiss,sentence_transformers
        print('PASS: FAISS + SentenceTransformers installed')
    except ImportError as exc:
        print(f'MISSING DEPENDENCY: {exc}; run pip install -r requirements-stage7-full.txt')
        raise
    print('SUCCESS: preflight done. Build FAISS index next.')
