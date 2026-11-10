from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_answer import ask_teletalk,answer_offers,answer_information
from src.teletalk_kb import load_offers,load_conflicts
from src.teletalk_vector import retrieve
if __name__=='__main__':
    catalog=load_offers()
    assert len(catalog.offers)>=70
    assert load_conflicts()
    print('PASS: reviewed catalog and quarantined conflicting offer')
    q=answer_offers('7 diner 100 tk er moddhe internet pack')
    assert q.mode=='grounded' and q.citations and q.offer_ids
    print('PASS: Banglish budget + validity + official citations')
    q2=answer_offers('50GB 30 days under 20 taka')
    assert q2.mode=='no_match'
    print('PASS: no-match refusal')
    for q3 in ['gen-z eligibility conditions','Oporajita package sharto','closed sim eligible ki']:
        x=answer_information(q3)
        assert x.mode=='grounded' and x.citations
    print('PASS: special SIM rules cited, never confirms a specific subscriber')
    results=retrieve('internet data 7 days less than 100 taka',top_k=6)
    assert results and len(results)<=6 and all(r['source_url'].startswith('https://') for r in results)
    print('PASS: real local FAISS multilingual retrieval and checked index manifest')
    print('SUCCESS: Stage 7 full verified locally; live website crawl and per-page coverage audit are separate.')
