"""Offline regression verifier and optional live smoke; does not invent a successful fetch."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_live import answer_live
from scripts.preflight_stage7_live import main as preflight


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check-live',action='store_true',help='Actually fetch official website; requires network and permission')
    a=p.parse_args()
    preflight()
    print('PASS: source guards and offline validation')
    if a.check_live:
        result=answer_live('100 takar moddhe 7 diner internet pack')
        print(result.message)
        print('MODE:',result.mode)
        if result.mode not in ('live_public_listing','no_match'):
            raise SystemExit('FAIL: official website could not be verified. Check network/robots/markup.')
        print('PASS: actual live fetch and extraction on this machine')
    else:
        print('NOTE: run --check-live to actually verify official network access')
    print('SUCCESS: Stage 7 live verifier completed')

if __name__=='__main__':main()
