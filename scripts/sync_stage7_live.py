"""Bounded website discovery + coverage report, never automatic commercial approval."""
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_discovery import discover
from src.teletalk_coverage import coverage
from src.teletalk_kb import STORE


def main(argv=None):
    p=argparse.ArgumentParser(description='Official sitemap discovery, audit-only')
    p.add_argument('--max-pages',type=int,default=100)
    p.add_argument('--delay',type=float,default=0.8)
    a=p.parse_args(argv)
    result=discover(max_pages=a.max_pages,delay=a.delay)
    current=coverage()
    path=STORE/'live_coverage_audit.json'
    path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Official public pages attempted:',result['attempted'])
    print('Captured for review:',result['captured'])
    print('Failed or disallowed:',len(result['failed']))
    print('Unvisited frontier:',result['frontier_left'])
    print('Source audit:',path)
    print('All newly discovered pages remain PENDING REVIEW, never auto approved.')
    print('Fresh source access:', 'SUCCESS (some official pages captured)' if result['captured'] else 'FAILED (no pages captured)')
    if result['failed']:
        print('Inspect failures in knowledge/stage7_full/site_discovery_report.json')
    return 0 if result['captured'] else 2

if __name__=='__main__':sys.exit(main())
