"""Polite, official-host-only page discovery with *pending review* quarantine.

Network fetches never add commercial claims to the approved answer/index automatically.
"""
from __future__ import annotations
from urllib.parse import urljoin,urlsplit,urlunsplit
from urllib.robotparser import RobotFileParser
from pathlib import Path
from datetime import date,datetime,timezone
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
import hashlib,json,time,re,requests
from src.teletalk_live import OfficialFetcher,LiveSourceError
from src.teletalk_kb import STORE,official_url,REVIEWED,load_offers,load_info

QUEUE=STORE/'review_queue'
REPORT=STORE/'site_discovery_report.json'
SEEDS=['https://www.teletalk.com.bd/en/sitemap','https://www.teletalk.com.bd/en',
       'https://www.teletalk.com.bd/internet-offers','https://www.teletalk.com.bd/voice',
       'https://www.teletalk.com.bd/vas','https://www.teletalk.com.bd/explore',
       'https://www.teletalk.com.bd/en/voice/prepaid/teletalk-professional-package',
       'https://www.teletalk.com.bd/offer/combo-package',
       'https://www.teletalk.com.bd/voice/prepaid/oporajita',
       'https://www.teletalk.com.bd/voice/prepaid/mayer-hasi']
FORBIDDEN=('login','logout','signup','sign-up','register','purchase','buy','checkout','payment','cart',
           'api/','my-account','order','online-recharge','bills/pay')
HEADERS={'User-Agent':'TeleRetainAI-Internship-Research/1.0 (public-docs-crawler; respectful, review-only)'}

def normalize_url(url:str)->str|None:
    try:
        p=urlsplit(url)
        u=urlunsplit(('https',p.netloc.lower(),p.path.rstrip('/') or '/', '', ''))
        official_url(u)
        if any(x in p.path.lower() for x in FORBIDDEN):return None
        if re.search(r'\.(jpg|png|gif|webp|svg|zip|pdf|docx?|xlsx?|js|css)$',p.path,re.I):return None
        return u
    except (ValueError,TypeError):return None

def extract(html:str,url:str)->dict:
    soup=BeautifulSoup(html,'html.parser')
    for tag in soup(['script','style','noscript','svg','iframe','form']):tag.decompose()
    for sel in ('header','footer','nav','.footer','.navbar','.site-footer'):
        for tag in soup.select(sel):tag.decompose()
    main=soup.select_one('main') or soup.select_one('article') or soup.select_one('.main-content') or soup.body or soup
    title=(soup.title.get_text(' ',strip=True) if soup.title else url)[:180]
    raw=main.get_text('\n',strip=True)
    text='\n'.join(line.strip() for line in raw.splitlines() if line.strip())[:80000]
    links=[]
    for a in main.find_all('a',href=True):
        link=normalize_url(urljoin(url,a['href']))
        if link and link not in links:links.append(link)
    # table extracts preserve rows for manual review (including parent row carry-forward).
    tables=[]
    for table in main.find_all('table'):
        rows=[]
        for tr in table.find_all('tr'):
            cells=[el.get_text(' ',strip=True)[:300] for el in tr.find_all(['th','td'],recursive=False)]
            if cells:rows.append(cells)
        if rows:tables.append(rows[:500])
    return dict(url=url,title=title,text=text,links=links,tables=tables)

def discover(*,max_pages:int=140,delay:float=.8,session=None,robots=None,fetcher=None)->dict:
    """Bounded public discovery. Pages remain PENDING REVIEW, never self-approved.

    Production uses the same OfficialFetcher, robots checks, redirect rules and
    verified Schannel fallback as live offers. An injected session+robots is
    supported solely for deterministic unit tests and controlled integrations.
    """
    if not 1<=max_pages<=500 or delay<.4:
        raise ValueError('Page limit 1–500, delay >=0.4 seconds.')
    if fetcher is not None and session is not None:
        raise ValueError('Pass fetcher OR test session, never both.')
    if session is not None and robots is None:
        raise ValueError('Injected test session requires explicit robots policy.')
    if fetcher is None and session is None:
        fetcher=OfficialFetcher(min_interval=delay,robots_override=robots)
    q=SEEDS.copy();seen=set();captures=[];failed=[]
    QUEUE.mkdir(parents=True,exist_ok=True)
    while q and len(seen)<max_pages:
        url=normalize_url(q.pop(0))
        if not url or url in seen:continue
        seen.add(url)
        try:
            if fetcher is not None:
                live=fetcher.fetch(url)  # Verified HTTPS & origin-specific robots.
                final=normalize_url(live.url)
                if not final:raise ValueError('Official URL check failed')
                payload=extract(live.html,final)
                stamp=live.fetched_at_utc
            else:
                # Explicit test session; NEVER the default production path.
                if not robots.can_fetch(HEADERS['User-Agent'],url):
                    failed.append({'url':url,'reason':'robots_disallowed'});continue
                response=session.get(url,headers=HEADERS,timeout=(5,18),allow_redirects=False,stream=True)
                try:
                    response.raise_for_status()
                    final=normalize_url(response.url)
                    if not final:raise ValueError('Redirected outside allowed site')
                    if 'text/html' not in response.headers.get('Content-Type','').lower():
                        failed.append({'url':url,'reason':'not_html'});continue
                    raw=bytearray()
                    for chunk in response.iter_content(32768):
                        raw.extend(chunk)
                        if len(raw)>1_200_000:raise ValueError('Page size limit exceeded')
                    payload=extract(bytes(raw).decode(response.encoding or 'utf-8',errors='replace'),final)
                    stamp=datetime.now(timezone.utc).isoformat()
                finally:
                    if callable(getattr(response,'close',None)):response.close()
            if len(payload['text'])<40:raise ValueError('No substantive page text')
            payload['sha256']=hashlib.sha256(payload['text'].encode('utf-8')).hexdigest()
            payload['fetched_at_utc']=stamp
            slug=hashlib.sha256(final.encode('utf-8')).hexdigest()[:20]
            (QUEUE/(slug+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            captures.append({'url':final,'id':slug,'sha256':payload['sha256'],'title':payload['title']})
            for link in payload['links'][:500]:
                if link not in seen and link not in q:q.append(link)
        except (requests.RequestException,LiveSourceError,ValueError,UnicodeError,OSError) as exc:
            failed.append({'url':url,'reason':type(exc).__name__+': '+str(exc)[:180]})
        if fetcher is None:time.sleep(delay)  # OfficialFetcher paces its own GETs.
    report={'attempted':len(seen),'captured':len(captures),'failed':failed,
            'pending_review':captures,'frontier_left':len(q),'limit_hit':bool(q),
            'complete':not q and not failed,
            'note':'Captured pages are UNVERIFIED PENDING REVIEW; cannot be served or recommended.'}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def approve_page(source_file:Path,*,expected_sha256:str,reviewer_acknowledgement:bool=False):
    """Reviewer must inspect saved text AND provide observed SHA; no auto approval."""
    if not reviewer_acknowledgement:raise ValueError('Manual review acknowledgement required.')
    source_file=Path(source_file).resolve()
    if source_file.parent!=QUEUE.resolve() or source_file.suffix!='.json':
        raise ValueError('Only captured review_queue pages may be approved.')
    p=json.loads(source_file.read_text(encoding='utf-8'))
    official_url(p['url'])
    if p['sha256']!=expected_sha256 or hashlib.sha256(p['text'].encode()).hexdigest()!=expected_sha256:
        raise ValueError('SHA mismatch: do not approve altered page.')
    from src.teletalk_kb import InfoPage
    body=p['text'][:9500]
    entry=dict(id='site-'+hashlib.sha256(p['url'].encode()).hexdigest()[:15],title=p['title'][:160],
               category='reviewed_site_page',source_url=p['url'],checked_on=datetime.now(ZoneInfo("Asia/Dhaka")).date().isoformat(),body=body)
    InfoPage.model_validate(entry)
    raw=json.loads(REVIEWED.read_text(encoding='utf-8')) if REVIEWED.exists() else {'schema_version':1,'pages':[]}
    raw['pages']=[x for x in raw['pages'] if x['id']!=entry['id']]+[entry]
    temp=REVIEWED.with_suffix('.tmp');temp.write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(REVIEWED)
    archive=source_file.parent.parent/'reviewed_source_archive'
    archive.mkdir(parents=True,exist_ok=True)
    destination=archive/source_file.name
    if destination.exists(): destination=archive/(source_file.stem+'_'+p['sha256'][:8]+'.json')
    source_file.replace(destination)
    return entry
