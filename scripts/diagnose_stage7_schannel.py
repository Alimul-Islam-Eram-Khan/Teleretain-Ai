"""Check verified Windows Schannel GET and real Stage 7 extraction; no API key."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.teletalk_schannel import SchannelTransportError, fetch_via_schannel
from src.teletalk_live import OFFERS_URL, OfficialFetcher, parse_live_offers, HEADERS
from bs4 import BeautifulSoup

def main():
    print('Stage 7 Windows Schannel HTTPS diagnostic (verified GET, no SSL bypass)')
    print('Configured fallback:', os.environ.get('TELERETAIN_SCHANNEL_FALLBACK', '(not enabled)'))
    for url in ('https://www.teletalk.com.bd/robots.txt', OFFERS_URL):
        try:
            r = fetch_via_schannel(url, headers=HEADERS)
            print('GET', url, '| HTTP', r.status_code, '| bytes', len(r.body),
                  '| MIME:', r.headers.get('Content-Type', '(missing)'))
            if '/robots.txt' in url and r.status_code == 404:
                print('NOTE: 404 on GET means no published robots.txt at that path; site terms still apply.')
        except SchannelTransportError as exc:
            print('FAILED strict Schannel GET:', str(exc)[:300]); return 1
    if os.environ.get('TELERETAIN_SCHANNEL_FALLBACK') != '1':
        print('NEXT: export TELERETAIN_SCHANNEL_FALLBACK=1')
        return 0
    try:
        fetcher = OfficialFetcher()
        page = fetcher.fetch(OFFERS_URL)
        offers, excluded = parse_live_offers(page)
        print('STAGE7 PAGE PASS:', page.url, '| parsed offers:',len(offers),
              '| excluded:',len(excluded))
        if not offers:
            soup = BeautifulSoup(page.html,'html.parser')
            tables = soup.find_all('table')
            print('DEBUG: HTML table count:', len(tables))
            for table in tables[:4]:
                cells = [x.get_text(' ',strip=True)[:70] for x in table.find_all(['th','td'],limit=6)]
                print('DEBUG: sample cells (official public page only):',cells)
            print('DEBUG: extraction exclusions:', [x.get('reason') for x in excluded[:8]])
            print('FAIL: live page downloaded but parsing produced zero safely validated offers.')
            return 1
        return 0
    except Exception as exc:
        print('STAGE7 FETCH/PARSE FAILED:', type(exc).__name__, str(exc)[:350])
        return 1

if __name__=='__main__': raise SystemExit(main())
