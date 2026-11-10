"""Diagnose Teletalk official-site access without exposing API keys or user data.
Does NOT bypass robots, redirects, access-denied responses or paid/private endpoints.
Run: python scripts/diagnose_stage7_live.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.teletalk_live import OfficialFetcher, LiveSourceError, OFFERS_FALLBACK_URLS, parse_live_offers, _access_restricted


def main():
    fetcher = OfficialFetcher()
    print('Stage 7 official website diagnostic (no API key, no customer data)')
    print('Checks: official HTTPS + robots.txt + HTML parse. No bypass on denied access.')
    success = False
    for url in OFFERS_FALLBACK_URLS:
        print('\nCHECK:', url)
        try:
            page = fetcher.fetch(url)
            offers, skipped = parse_live_offers(page)
            print('FETCH: PASS | source:', page.url)
            print('PAGE: title=', page.title[:90], '| text characters=', len(page.text))
            print('PARSE: verified offer rows=',len(offers),'| excluded=',len(skipped))
            print('FETCHED UTC:',page.fetched_at_utc)
            if offers:
                success = True
                break
            print('STATUS: HTML available, parser did not accept any safe offer rows.')
        except LiveSourceError as exc:
            print('FETCH: FAIL | reason:', str(exc)[:1000])
            if _access_restricted(exc):
                print('STATUS: access denied by official site/policy; stop (no host rotation).')
                break
    if success:
        print('\nSUCCESS: live verified official offer rows are retrievable.')
        return 0
    print('\nFAIL: Could not verify official offer data. No old prices should be displayed as live.')
    print('If browser works but requests fail, report the reason above (403, 429, DNS, SSL, robots, 502, or parse).')
    return 1

if __name__=='__main__':
    sys.exit(main())
