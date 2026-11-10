"""Offline preflight: source guard, deps, snapshot separation; not a live connectivity claim."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_live import official, OFFERS_URL, SITEMAP_URL, _valid_days, _int_amount, LiveSourceError


def main():
    import requests,bs4,pydantic
    assert official(OFFERS_URL)==OFFERS_URL
    assert official(SITEMAP_URL)==SITEMAP_URL
    for bad in ('https://evil.example/','http://www.teletalk.com.bd/internet-offers',
                'https://www.teletalk.com.bd.evil.example/a',
                'https://www.teletalk.com.bd/login',
                'https://www.teletalk.com.bd/internet-offers?customer_id=123'):
        try:official(bad)
        except LiveSourceError:continue
        raise AssertionError(f'Unsafe URL accepted: {bad}')
    assert _int_amount('TK 56')==56
    assert _valid_days('7 Days')==7
    print('PASS: requests, BeautifulSoup, Pydantic available')
    print('PASS: official HTTPS allowlist and forbidden URLs')
    print('PASS: sample offer text conversion')
    print('SUCCESS: Stage 7 live preflight (network NOT tested)')

if __name__=='__main__': main()
