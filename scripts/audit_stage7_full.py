from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_coverage import coverage
from src.teletalk_kb import STORE
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    r=coverage()
    print('Approved records:',r['approved_offers'],' Reviewed pages:',r['reviewed_information_entries'])
    print('Categories:',r['offer_categories'])
    print('Review queue:',r['review_queue_count'])
    print('Conflicts quarantined:',len(r['quarantined_conflicts']))
    print('DISCOVERY:', 'not run' if not r['discovery'] else f"{r['discovery']['captured']} captured / {r['discovery']['attempted']} attempted")
    print('WARNING: 100% current website coverage is NOT certified.')
    if args.write:
        path=STORE/'coverage_audit.json';path.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Written:',path)
