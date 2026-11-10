from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_discovery import discover
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Respectful public Teletalk website discovery; creates UNAPPROVED queue only')
    parser.add_argument('--max-pages',type=int,default=140)
    parser.add_argument('--delay',type=float,default=.8)
    args=parser.parse_args()
    result=discover(max_pages=args.max_pages,delay=args.delay)
    print(f"Captured {result['captured']} official pages; attempted {result['attempted']}; pending review only.")
    print('Failures:',len(result['failed']),' Frontier remaining:',result['frontier_left'])
    print('See knowledge/stage7_full/site_discovery_report.json and review_queue/')
