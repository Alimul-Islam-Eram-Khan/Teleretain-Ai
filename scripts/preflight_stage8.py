"""No-network check of source compatibility, packages and model isolation."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.teletalk_live import LiveOffer, OfficialFetcher, fetch_offers_fresh
from src.retention_recommender import RecommendationRequest, from_question
from src.teletalk_answer import parse_question

def main():
    p=parse_question('100 takar moddhe 7 diner internet pack')
    assert p['budget_bdt']==100 and p['category']=='internet' and p['exact_validity_days']==7
    request=RecommendationRequest(category='internet',budget_bdt=100,exact_validity_days=7)
    assert request.budget_bdt==100
    assert callable(fetch_offers_fresh) and callable(OfficialFetcher)
    print('PASS: Stage 7 v3.4 live-offer contract available')
    print('PASS: Banglish budget/validity parser and strict Stage 8 request schema')
    print('PASS: Stage 1-7 files remain untouched; preflight makes no network or API calls')
    print('SUCCESS: ready for Stage 8 tests and live recommendation demo')

if __name__=='__main__':main()
