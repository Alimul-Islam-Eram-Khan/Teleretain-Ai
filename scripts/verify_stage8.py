"""Stage 8 acceptance: offline tests by default, opt-in live official site test.

Run with TELERETAIN_SCHANNEL_FALLBACK=1 on the user's validated Windows setup.
"""
import sys,subprocess,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-live',action='store_true',help='Fetch current official HTML and verify 4 package categories')
    args=ap.parse_args()
    if not args.check_live:
        print('Stage 8 offline recommendation acceptance (NO network)')
        files = sorted(str(p) for p in (root/'tests').glob('test_stage8_*.py'))
        if not files:
            print('FAIL: no Stage 8 tests present'); return 1
        p=subprocess.run([sys.executable,'-m','pytest','-q',*files],check=False)
        return p.returncode
    from src.teletalk_live import OfficialFetcher, fetch_offers_fresh
    from src.retention_recommender import RecommendationRequest, recommend
    from src.teletalk_live import LiveSourceError
    fetcher=OfficialFetcher()
    try:
        page,offers,excluded=fetch_offers_fresh(fetcher)
    except (LiveSourceError, OSError) as exc:
        print(f'FAIL: official live source unavailable: {type(exc).__name__}: {exc}')
        return 1
    print('Live official page:',page.url,'| UTC:',page.fetched_at_utc,'| parsed offers:',len(offers))
    def local_verified_page(_):return page,offers,excluded
    counts=[]
    for category in ('internet','combo','minute','sms'):
        result=recommend(RecommendationRequest(category=category,limit=3),fetcher=fetcher,offer_loader=local_verified_page)
        ok=result.status=='recommended' and all(x.source_url==page.url for x in result.items)
        print(f'{"PASS" if ok else "FAIL"}: {category} | {result.status} | cited={bool(result.citations)}')
        counts.append(ok)
    if not all(counts):return 1
    print('SUCCESS: Stage 8 actual official-source matching verified on this machine.')
    print('Note: only this public page was checked; no 100% catalog, SIM eligibility, or real-world churn uplift claim.')
    return 0
if __name__=='__main__':sys.exit(main())
