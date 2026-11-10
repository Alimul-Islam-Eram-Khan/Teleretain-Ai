from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_discovery import approve_page,QUEUE
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Manual review-only approval. Review text and SHA yourself first.')
    p.add_argument('--file',required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--i-reviewed-official-page',action='store_true')
    args=p.parse_args()
    path=Path(args.file)
    result=approve_page(path,expected_sha256=args.sha256,reviewer_acknowledgement=args.i_reviewed_official_page)
    print('APPROVED INFO PAGE ONLY (not offer price):',result['title'])
    print('Source:',result['source_url'])
    print('Rebuild FAISS: python scripts/build_stage7_full.py')
