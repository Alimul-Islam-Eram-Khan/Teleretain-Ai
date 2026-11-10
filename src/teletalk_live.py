"""Live public Teletalk reader. No account endpoints, no private identifiers, no LLM.

Only freshly fetched official HTML may support a *current public listing* response.
A live listing never proves a particular SIM can activate the offer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from collections import defaultdict
from bs4 import BeautifulSoup
import hashlib
import json
import re
import time
import unicodedata
import requests

# Load only locally configured opt-in flags, without overwriting terminal values.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
except ImportError:
    pass

from src.teletalk_answer import KnowledgeAnswer, parse_question, SIM_ALIASES
from src.teletalk_kb import STORE, load_info
from src.teletalk_tls import official_session
from src.teletalk_schannel import SchannelTransportError, fetch_via_schannel

BASE = 'https://www.teletalk.com.bd'
OFFERS_URL = BASE + '/internet-offers'
# Alternate public document endpoints on Teletalk's two official hostnames.
# These are used for availability/URL differences, never to bypass robots/403/429.
OFFERS_FALLBACK_URLS = (
    OFFERS_URL,
    'https://teletalk.com.bd/internet-offers',
    'https://teletalk.com.bd/bn/internet-offers',
    'https://www.teletalk.com.bd/bn/internet-offers',
)
SITEMAP_URL = BASE + '/en/sitemap'
USER_AGENT = 'TeleRetainAI-PublicAcademicDemo/1.0 (respectful official-page research)'
MAX_BYTES = 2_000_000
MAX_LINKS = 300
AUDIT_DIR = STORE / 'live_audit'
EXCLUDE_PATH = ('/login', '/logout', '/signup', '/sign-up', '/register', '/purchase', '/checkout',
                '/payment', '/cart', '/my-account', '/api/', '/online-recharge', '/order')
BLOCK_EXT = ('.pdf', '.zip', '.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp', '.exe', '.css', '.js')
HEADERS = {'User-Agent': USER_AGENT, 'Accept': 'text/html,application/xhtml+xml'}

class LiveSourceError(RuntimeError):
    """Never silently replace a current official page with an old price snapshot."""


def digits(s: str) -> str:
    return ''.join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in s)


def official(url: str) -> str:
    if not isinstance(url, str) or len(url) > 400:
        raise LiveSourceError('Invalid source URL')
    p = urlsplit(url)
    if p.scheme != 'https' or p.hostname not in {'www.teletalk.com.bd', 'teletalk.com.bd'} or (
        p.username or p.password or p.port or p.query or p.fragment or not p.path.startswith('/')
    ):
        raise LiveSourceError('Official Teletalk HTTPS documents only')
    path = p.path.lower()
    if any(x in path for x in EXCLUDE_PATH) or path.endswith(BLOCK_EXT):
        raise LiveSourceError('Non-public or unsupported path blocked')
    return urlunsplit(('https', p.hostname, p.path, '', ''))


def _clean_text(tag) -> str:
    return re.sub(r'\s+', ' ', tag.get_text(' ', strip=True)).strip()


@dataclass(frozen=True)
class LivePage:
    url: str
    html: str
    text: str
    title: str
    fetched_at_utc: str
    sha256: str


class OfficialFetcher:
    """Bounded official-only HTTPS fetch, fail-closed robots and strict redirects."""
    def __init__(self, session=None, *, min_interval: float = 0.5, clock=time.monotonic,
                 sleep=time.sleep, robots_override=None):
        if min_interval < 0.4:
            raise ValueError('Request interval must be >= 0.4 s')
        self.session = session if session is not None else official_session()
        self._is_default_session = session is None
        self.min_interval = min_interval
        self.clock, self.sleep = clock, sleep
        self.last_request = None
        self.robots = robots_override

    def _pace(self):
        now = self.clock()
        if self.last_request is not None:
            remaining = self.min_interval - (now - self.last_request)
            if remaining > 0: self.sleep(remaining)
        self.last_request = self.clock()

    def _raw(self, url: str):
        self._pace()
        try:
            return self.session.get(url, headers=HEADERS, timeout=(5, 18),
                                    stream=True, allow_redirects=False)
        except requests.exceptions.SSLError as exc:
            # Explicit opt-in; never used for injected/mock sessions. Curl must
            # verify TLS with Schannel, not --insecure/verify=False.
            if (self._is_default_session and
                    __import__('os').environ.get('TELERETAIN_SCHANNEL_FALLBACK') == '1'):
                try:
                    return fetch_via_schannel(url, headers=HEADERS, max_bytes=MAX_BYTES)
                except SchannelTransportError as curl_exc:
                    raise LiveSourceError('Official HTTPS failed in Requests and strict '
                                          f'Schannel fallback: {curl_exc}') from curl_exc
            raise LiveSourceError('Official source unavailable: SSLError. '
                                  'Optional verified Schannel fallback is not enabled.') from exc
        except requests.RequestException as exc:
            raise LiveSourceError(f'Official source unavailable: {type(exc).__name__}') from exc

    def _get_robots(self, host: str):
        if self.robots is not None and not isinstance(self.robots,dict): return self.robots
        if isinstance(self.robots,dict) and host in self.robots: return self.robots[host]
        # Robots rules are origin-specific; an alternate Teletalk hostname has its own.
        url = 'https://' + host + '/robots.txt'
        for _ in range(3):
            response = self._raw(url)
            try:
                if response.status_code in (301,302,303,307,308):
                    url = official(urljoin(url,response.headers.get('Location','')))
                    continue
                if response.status_code == 404:
                    text = ''  # A missing robots file imposes no extra rules.
                elif response.status_code == 200:
                    text = self._read_bytes(response,max_size=100_000).decode('utf-8',errors='replace')
                else:
                    raise LiveSourceError(f'Cannot establish robots policy (HTTP {response.status_code})')
                rules = RobotFileParser(); rules.parse(text.splitlines())
                if not isinstance(self.robots,dict):self.robots = {}
                self.robots[host] = rules
                return rules
            finally:
                response.close()
        raise LiveSourceError('Too many robots policy redirects')

    @staticmethod
    def _read_bytes(response, *, max_size=MAX_BYTES) -> bytes:
        # Never trust Content-Length or download unlimited payloads.
        if int(response.headers.get('Content-Length', '0') or '0') > max_size:
            raise LiveSourceError('Official page exceeds size limit')
        result = bytearray()
        for chunk in response.iter_content(chunk_size=32_768):
            result.extend(chunk)
            if len(result) > max_size:
                raise LiveSourceError('Official page exceeds size limit')
        return bytes(result)

    def fetch(self, url: str) -> LivePage:
        target = official(url)
        for _ in range(3):
            rules = self._get_robots(urlsplit(target).hostname)
            if not rules.can_fetch(USER_AGENT, target):
                raise LiveSourceError('Official robots.txt disallows this page')
            response = self._raw(target)
            try:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location: raise LiveSourceError('Redirect without Location')
                    target = official(urljoin(target, location))
                    continue
                if response.status_code != 200:
                    raise LiveSourceError(f'Official page returned HTTP {response.status_code}')
                mime = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in mime:
                    raise LiveSourceError('Expected official HTML; received another content type')
                raw = self._read_bytes(response)
                enc = response.encoding or 'utf-8'
                if enc.lower() in ('iso-8859-1', 'latin-1'):
                    enc = 'utf-8'  # Modern Teletalk pages with Bangla content use UTF-8.
                html = raw.decode(enc, errors='replace')
                soup = BeautifulSoup(html, 'html.parser')
                for el in soup(['script', 'style', 'noscript', 'svg', 'iframe', 'form']): el.decompose()
                content = soup.select_one('main') or soup.select_one('article') or soup.body or soup
                text = '\n'.join(x.strip() for x in content.get_text('\n', strip=True).splitlines() if x.strip())
                if len(text) < 30:
                    raise LiveSourceError('Official page has too little readable text (may need JavaScript)')
                stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
                return LivePage(target, html, text[:160_000],
                                _clean_text(soup.title)[:180] if soup.title else target,
                                stamp, hashlib.sha256(raw).hexdigest())
            finally:
                response.close()
        raise LiveSourceError('Too many official redirects')


@dataclass(frozen=True)
class LiveOffer:
    category: str
    description: str
    price_bdt: int
    validity_days: int | None
    ussd: str
    source_url: str
    fetched_at_utc: str
    evidence: str
    sim_eligibility: str = 'unverified_for_individual'

    def key(self):
        return self.ussd


def _int_amount(s: str) -> int | None:
    normalized = digits(s).replace(',', '')
    m = re.search(r'(?:\btk\.?\s*|\bbdt\s*|৳\s*|টাকা\s*)(\d{1,5})(?:\.00)?\b', normalized, flags=re.I)
    return int(m.group(1)) if m else None


def _valid_days(s: str) -> int | None:
    t = digits(s).lower()
    if re.search(r'(?:unlimited|no expiry|আজীবন)', t): return None
    m = re.search(r'(\d{1,3})\s*(?:days?|দিন|diner|din)\b', t)
    if m: return int(m.group(1))
    m = re.search(r'(\d{1,2})\s*(?:hours?|hrs?|ঘন্টা|ঘণ্টা)', t)
    if m: return 1 if int(m.group(1)) == 24 else None
    return None


def _category(text: str) -> str | None:
    t = text.casefold()
    if re.search(r'combo|কম্বো', t): return 'combo'
    if re.search(r'minute|মিনিট', t): return 'minute'
    if re.search(r'sms|এসএমএস', t): return 'sms'
    if re.search(r'unlimited|weekend|uninterrupted|আনলিমিটেড|উইকেন্ড|নিরবচ্ছিন্ন', t): return 'unlimited_weekend'
    if re.search(r'data|internet|ডাটা|ডেটা|ইন্টারনেট', t): return 'internet'
    return None


def _category_from_offer_description(description: str) -> str | None:
    '''Infer only when the row itself contains unambiguous bundle terms.'''
    d = description.casefold()
    if 'weekend' in d or 'unlimited' in d or 'uninterrupted' in d or 'উইকেন্ড' in d:
        return 'unlimited_weekend'
    if re.search(r'minute_bundle|মিনিট_বান্ডেল',d):return 'minute'
    if re.search(r'sms_bundle',d):return 'sms'
    has_data = bool(re.search(r'\d+\s*(?:\.\d+)?\s*(?:gb|mb|জিবি|এমবি)',d))
    has_voice = bool(re.search(r'\d+\s*(?:minutes?|min|মিনিট)',d))
    has_sms = bool(re.search(r'\d+\s*(?:sms|এসএমএস)',d))
    if has_data and (has_voice or has_sms):return 'combo'
    if has_data:return 'internet'
    return None


def _find_code(text: str) -> str | None:
    text = digits(text)
    m = re.search(r'\*[\d*]{2,25}#', text)
    return m.group(0) if m else None


def _offer_table_rows(table) -> list[list[str]]:
    out = []
    for tr in table.select('tr'):
        cells = [_clean_text(c) for c in tr.find_all(['th', 'td'], recursive=False)]
        if cells: out.append(cells)
    return out


def parse_live_offers(page: LivePage) -> tuple[list[LiveOffer], list[dict]]:
    """Conservative table extraction: never infer missing prices/codes from nearby prose.

    Only regular public Internet Offers page is treated as general-public catalog.
    Special-SIM pages require eligibility verification and are handled as information.
    """
    if urlsplit(page.url).path.lower().rstrip('/') not in {'/internet-offers', '/en/internet-offers', '/bn/internet-offers'}:
        return [], []
    soup = BeautifulSoup(page.html, 'html.parser')
    collected, excluded = [], []
    for table in soup.find_all('table'):
        rows = _offer_table_rows(table)
        if not rows: continue
        # Section heading nearest before table; heading row may be inside it.
        ancestor = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading = _clean_text(ancestor) if ancestor else ''
        if table.caption: heading += ' ' + _clean_text(table.caption)
        category = _category(heading)
        inherited_description = ''
        group_days = None
        for cells in rows:
            joined = ' | '.join(cells)
            if len(cells) == 1:
                if _category(cells[0]):
                    category = _category(cells[0]); inherited_description = ''; group_days = None
                elif _valid_days(cells[0]): group_days = _valid_days(cells[0])
                continue
            # May be repeated header rows of the different sections.
            if any('USSD' in c.upper() or 'ইউএসএসডি' in c for c in cells) and any(
                'price' in c.lower() or 'মূল্য' in c for c in cells
            ):
                if _category(joined): category = _category(joined)
                continue
            if _category(joined) and 'package' in joined.lower() and not _find_code(joined):
                category = _category(joined); inherited_description = ''; group_days = None
                continue
            code = _find_code(joined)
            price_cells = [c for c in cells if _int_amount(c) is not None]
            if not code or len(price_cells) != 1: continue
            price = _int_amount(price_cells[0])
            # An explicit validity cell must be distinct from the package volume cell.
            validity = next((_valid_days(c) for c in cells if _valid_days(c) is not None), None)
            validity = validity or group_days
            # Description can carry forward for compact 4-cell continuation combo rows.
            first = cells[0]
            continuation = _valid_days(first) is not None or _int_amount(first) is not None
            if continuation and inherited_description:
                description = inherited_description
            else:
                description = first
                if not continuation: inherited_description = description
            category = _category_from_offer_description(description) or category or _category(heading)
            if not category:
                excluded.append({'reason': 'unknown_section', 'evidence': joined[:240]}); continue
            # Guard against a table being silently assigned to the previous section.
            desc_lower = description.casefold()
            cues = {'internet': ('gb', 'mb', 'জিবি', 'এমবি'),
                    'combo': ('minute', 'min', 'মিনিট'),
                    'minute': ('minute', 'min', 'মিনিট'),
                    'sms': ('sms', 'এসএমএস')}
            if category in cues and not any(t in desc_lower for t in cues[category]):
                excluded.append({'reason':'category_description_mismatch','evidence':joined[:240]}); continue
            if not description or not (0 < price <= 100_000) or len(description) > 170:
                excluded.append({'reason': 'invalid_description_or_price', 'evidence': joined[:240]}); continue
            # Every offer must have an original, inspectable table row.
            collected.append(LiveOffer(category, description, price, validity, code,
                                       page.url, page.fetched_at_utc, joined[:400]))
    # Source itself can contain contradictory codes/prices. Quarantine such codes.
    by_code = defaultdict(set)
    for o in collected:
        by_code[o.key()].add((o.category, o.description, o.price_bdt, o.validity_days))
    bad = {code for code, values in by_code.items() if len(values) > 1}
    for code in sorted(bad): excluded.append({'reason':'conflicting_same_page_offer', 'code':code})
    clean = []
    seen = set()
    for o in collected:
        if o.key() in bad: continue
        fingerprint = (o.category, o.description, o.price_bdt, o.validity_days, o.ussd)
        if fingerprint not in seen:
            seen.add(fingerprint); clean.append(o)
    return clean, excluded


def _filter_live(offer: LiveOffer, params: dict) -> bool:
    if params['budget_bdt'] is not None and offer.price_bdt > params['budget_bdt']:return False
    if params['min_validity_days'] is not None and (
        offer.validity_days is None or offer.validity_days < params['min_validity_days']
    ): return False
    if params['exact_validity_days'] is not None and offer.validity_days != params['exact_validity_days']: return False
    if params.get('ussd_code') and offer.ussd != params['ussd_code']:return False
    if params['category'] and params['category'] != offer.category:return False
    # Data/minutes/SMS quantities may be combined in one descriptive field.
    desc = digits(offer.description).lower().replace(' ', '')
    requirements = [('min_data_mb', r'(\d+(?:\.\d+)?)(gb|mb)', 1024),
                    ('min_minutes', r'(\d+)(?:minutes?|min)', 1),
                    ('min_sms', r'(\d+)sms', 1)]
    for key,pattern,_ in requirements:
        minimum = params.get(key)
        if minimum is None: continue
        found = re.findall(pattern, desc)
        if key == 'min_data_mb':
            values = [round(float(num) * (1024 if unit == 'gb' else 1)) for num,unit in found]
        else:
            values = [int(v) for v in found]
        if not values or max(values) < minimum: return False
    return True


def _snippet(text: str, question: str, limit=4, suppress_prices=False) -> list[str]:
    """Legacy text-only helper; no unrelated lines are used as filler."""
    clean = [re.sub(r'\s+', ' ', x).strip() for x in text.splitlines()]
    clean = [x for x in clean if 15 <= len(x) <= 700]
    if suppress_prices:
        clean = [x for x in clean if not re.search(r'৳|(?:tk\.?|taka|টাকা|price|মূল্য)\s*[:@]?', x, re.I)]
    words = [x for x in re.findall(r'[a-z\u0980-\u09ff]{3,}', question.casefold())
             if x not in {'what','the','please','explain','conditions','eligibility',
                          'information','teletalk','offer','sim','customer','care',
                          'আমার','আমাকে','শর্ত','কি','কিভাবে'}]
    rank = [(sum(w in line.casefold() for w in words), -i, line)
            for i,line in enumerate(clean)]
    return [line for score,_,line in sorted(rank,reverse=True) if score > 0][:limit]


# Source-only evidence: reject navigation menus and prefer actual eligibility,
# activation and service sentences from the current official page. No LLM used.
_NAV_NOISE = (
    'corporate information', 'management information', 'download myteletalk app',
    'online recharge & purchase', 'privacy policy', 'about us', 'useful links',
    'contact us', 'photo gallery', 'board of directors', 'tender notice',
    'feedback', 'copyright', 'site map', 'sitemap',
)
_ELIGIBILITY_TERMS = (
    'eligib', 'selected customer', 'only selected', 'must ', 'need to',
    'applicable', 'registration', 'register', 'verification', 'verify',
    'born between', 'new customer', 'existing customer', 'nid', 'btrc',
    'to check', 'check whether', 'to avail', 'dial ', 'activation',
    'validity', 'restriction', 'campaign period', 'once', 'carry forward',
    'not applicable', 'should', 'gpa', 'student', 'professional',
    'entitled', 'required', 'customer list', 'migration',
    'নির্বাচিত', 'গ্রাহক', 'উপযুক্ত', 'যোগ্য', 'নিবন্ধন', 'শর্ত',
    'রেজিস্ট্রেশন', 'জন্ম', 'শিক্ষার্থী', 'ডায়াল', 'ডায়াল',
    'অফারের আওতা', 'ক্যাম্পেইন', 'মেয়াদ', 'মেয়াদ',
)
_SERVICE_TERMS = (
    'help line', 'helpline', 'hotline', 'call center', 'customer care',
    'other operator', 'corporate', 'telecharge', 'dial ', 'contact',
    'passport', 'requirement', 'roaming', 'sim replacement', 'migration',
    'sms', 'helpline', 'হেল্পলাইন', 'হটলাইন', 'কল সেন্টার',
    'কাস্টমার কেয়ার', 'কাস্টমার কেয়ার', 'যোগাযোগ',
)


def _page_evidence(page: LivePage, question: str, *, special: bool, limit: int = 6) -> list[str]:
    """Read relevant sentences from official document blocks, not page navigation.

    A page may be HTTP 200 yet only contain menus. Such a response must NOT
    be treated as a successfully verified answer to the customer's question.
    """
    soup = BeautifulSoup(page.html, 'html.parser')
    for node in list(soup.select('script,style,noscript,iframe,form,nav,header,footer,aside,'
                                 '.navbar,.sidebar,.breadcrumb,.breadcrumbs,.site-footer,'
                                 '.footer,.menu,.social,.social-links')):
        node.decompose()
    root = soup.select_one('main') or soup.select_one('article') or soup.body or soup
    candidates = []
    # Extract blocks, including table rows for FAQs and contact information.
    for node in root.select('li,p,tr'):
        if node.find_parent(['nav','header','footer','aside']):
            continue
        value = _clean_text(node)
        if not 9 <= len(value) <= 850:
            continue
        if any(noise in value.casefold() for noise in _NAV_NOISE):
            continue
        if special and re.search(r'৳|(?:tk\.?|taka|টাকা|price|মূল্য)\s*[:@]?',value,re.I):
            continue
        if value not in candidates:
            candidates.append(value)
    # Some sites use divs rather than semantic p/li elements; rescue ONLY
    # relevant short leaf blocks, never a full nav-filled parent container.
    if not candidates:
        for node in root.select('div'):
            if node.find(['p','li','tr','div']):
                continue
            value = _clean_text(node)
            if 18 <= len(value) <= 450 and not any(x in value.casefold() for x in _NAV_NOISE):
                candidates.append(value)
    keys = _ELIGIBILITY_TERMS if special else _SERVICE_TERMS
    # A direct page for customer care can contain short phone labels.
    words = [w for w in re.findall(r'[a-z\u0980-\u09ff]{3,}',question.casefold())
             if w not in {'teletalk','customer','care','information','details','please',
                          'about','condition','conditions','eligibility','sim','the','for'}]
    scored = []
    for i,line in enumerate(candidates):
        text = line.casefold()
        topic = sum(token in text for token in keys)
        match = sum(w in text for w in words)
        is_contact = bool(re.search(r'\b121\b|\b267\b|0150\d{7}|help\s*line|helpline',text,re.I))
        is_rule = bool(re.search(r'(?:only selected|eligible|eligibility|to check|to avail|must|nid|customer list|migration|born between|\*155\*)',text,re.I))
        if special and not (topic or is_rule): continue
        if not special and not (topic or match or is_contact): continue
        # Favor actual conditions/contacts; de-prioritize generic paragraph headers.
        score = topic * 3 + match * 2 + (10 if special and is_rule else 0) + (10 if not special and is_contact else 0)
        scored.append((-score,i,line))
    scored.sort()
    return [line for _,_,line in scored[:limit]]


# URLs confirmed to be public Teletalk web pages. They are fetched fresh;
# these are source locators, never hardcoded prices, phone numbers or rules.
_DIRECT_SERVICES = {
    'call_center': 'https://www.teletalk.com.bd/en/customer-care-center/call-center',
    'care_centres': 'https://www.teletalk.com.bd/en/customer-care-center',
    'faq': 'https://www.teletalk.com.bd/en/others/faq',
    'roaming': 'https://www.teletalk.com.bd/en/voice/international-roaming',
}


def _select_source(question: str, *, fetcher: OfficialFetcher) -> str | None:
    q = question.casefold()
    if any(x in q for x in ('customer care','call center','call centre','helpline','help line','কাস্টমার কেয়ার','কল সেন্টার')):
        if any(x in q for x in ('address','location','nearest','center address','centre address','ঠিকানা','কোথায়')):
            return official(_DIRECT_SERVICES['care_centres'])
        return official(_DIRECT_SERVICES['call_center'])
    if any(x in q for x in ('faq','frequently asked','প্রশ্নোত্তর')):
        return official(_DIRECT_SERVICES['faq'])
    if any(x in q for x in ('roaming','রোমিং')):
        return official(_DIRECT_SERVICES['roaming'])
    info = load_info(allow_stale=True)
    for ident,aliases in SIM_ALIASES.items():
        if any(a in q for a in aliases):
            matches = [x for x in info if x.id == {'gen_z':'sim-gen-z','professional':'sim-professional',
                'closed_sim':'sim-closed','oporajita':'sim-oporajita','mayer_hasi':'sim-mayer-hasi',
                'agami':'sim-agami','bornomala':'sim-bornomala'}.get(ident)]
            if matches:return official(matches[0].source_url)
    direct = {'emergency':'service-emergency','জরুরি':'service-emergency','gravity':'postpaid-gravity',
              'postpaid':'postpaid-gravity','rate cutter':'service-rate-cutter','rate-cutter':'service-rate-cutter',
              'roaming':'service-roaming','customer care':'service-customer-care',
              'faq':'service-faq','teleplan':'service-teleplan','e-sim':'service-esim'}
    for token, ident in direct.items():
        if token in q:
            row = next((x for x in info if x.id == ident),None)
            if row:
                resolved=official(row.source_url)
                # A homepage/sitemap is a directory, not evidence of detailed conditions.
                if urlsplit(resolved).path.lower() not in ('/en/sitemap','/bn/sitemap','/en','/'):
                    return resolved
    # Last resort: discover exact official sitemap links by matching source URL slug.
    page = fetcher.fetch(SITEMAP_URL)
    soup = BeautifulSoup(page.html, 'html.parser')
    terms = [x for x in re.findall(r'[a-z\u0980-\u09ff]+',q) if len(x)>3]
    if not terms: return None
    scored = []
    for a in soup.select('a[href]')[:MAX_LINKS]:
        label = (_clean_text(a) + ' ' + a['href']).casefold()
        try: u = official(urljoin(page.url,a['href']))
        except LiveSourceError: continue
        score = sum(token in label for token in terms)
        if score >= 2: scored.append((score,u))
    return max(scored,default=(0,None))[1]


def _access_restricted(exc: LiveSourceError) -> bool:
    """Do not rotate hostnames to work around an explicit restriction."""
    error = str(exc)
    if 'robots.txt disallows' in error:
        return True
    return bool(re.search(r'HTTP (?:401|403|429)\b', error))


def fetch_offers_fresh(fetcher: OfficialFetcher) -> tuple[LivePage, list[LiveOffer], list[dict]]:
    """Try only official public paths, respecting robots and HTTP access restrictions.

    A parse failure on one legitimate public URL is not proof the others are
    unavailable. An access-denied response, however, is never worked around.
    """
    errors = []
    for url in OFFERS_FALLBACK_URLS:
        try:
            page = fetcher.fetch(url)
            offers, excluded = parse_live_offers(page)
            if offers:
                return page, offers, excluded
            errors.append(f'{url}: no safely parsed table rows')
        except LiveSourceError as exc:
            if _access_restricted(exc):
                raise LiveSourceError(f'Access policy prevents automatic fetch: {exc}') from exc
            errors.append(f'{url}: {exc}')
    raise LiveSourceError('All official public offer endpoints unavailable or unparseable: ' +
                          '; '.join(errors)[:1100])


def answer_live(question: str, *, fetcher: OfficialFetcher | None=None, limit: int=6,
                audit: bool=True) -> KnowledgeAnswer:
    """Deterministic, fresh official-page-backed response. Never uses an old price if live fetch fails."""
    label = 'Teletalk official public web page (fetched now; personal eligibility unverified)'
    if not isinstance(question,str) or not question.strip() or len(question)>600:
        return KnowledgeAnswer('unavailable',label,'Please provide a short public-offer question.')
    if not 1 <= limit <= 12: raise ValueError('Result limit must be 1–12')
    fetcher = fetcher or OfficialFetcher()
    last_source = OFFERS_URL
    try:
        params = parse_question(question)
        informational = params['info_request'] or bool(params['sim_type']) or any(
            t in question.casefold() for t in ('roaming','emergency','customer care','faq','e-sim','teleplan','gravity')
        )
        if informational:
            source = (OFFERS_URL if params['category'] and not params['sim_type']
                      else _select_source(question,fetcher=fetcher))
            if not source:
                return KnowledgeAnswer('unavailable',label,
                    'Cannot locate a matching official page. No eligibility or price claim is made.')
            last_source = source
            page = fetcher.fetch(source)
            # Special SIM price pages can contain contradictory advertised and card prices.
            # Never interpret quoted promotional snippets as a personalized eligible price.
            special = bool(params['sim_type'])
            fragments = _page_evidence(page,question,special=special)
            if not fragments:
                raise LiveSourceError('Official page contains no relevant, verifiable conditions or service details')
            msg = ('Official page excerpts (not personalized eligibility verification):\n' +
                   '\n'.join('• '+x for x in fragments) +
                   f'\nOfficial source: {page.url}\nFetched UTC: {page.fetched_at_utc}\n'
                   + ('Special SIM prices are NOT asserted from these excerpts; check current restrictions with Teletalk.\n' if special else '') +
                   'Confirm eligibility and activation with Teletalk; subscriber status is not accessible.')
            if audit: _audit_read(page,kind='information',count=len(fragments))
            return KnowledgeAnswer('live_source_information',label,msg,(page.url,),provider='official-website')
        page, extracted, excluded = fetch_offers_fresh(fetcher)
        if params['category'] is None and not any(t in question.casefold() for t in
            ('offer','package','প্যাক','অফার','দাম','price','ussd','tk','taka','টাকা')):
            return KnowledgeAnswer('needs_clarification',label,
                  'Please specify a package type (internet/combo/minute/SMS), budget or duration.')
        chosen = sorted((o for o in extracted if _filter_live(o,params)),
                        key=lambda o:(o.price_bdt,-(o.validity_days or 0),o.description))[:limit]
        if audit: _audit_read(page,kind='offer',count=len(extracted),skipped=excluded,offers=extracted)
        if not chosen:
            return KnowledgeAnswer('no_match',label,
                f'No matching explicitly priced and coded public listing was found in the freshly fetched page. '
                f'This does not prove that no other Teletalk offer exists.\nSource: {page.url}\nFetched UTC: {page.fetched_at_utc}',
                (page.url,),provider='official-website')
        output = ['Freshly fetched official *public listings*; no live activation or SIM-eligibility check:',]
        for o in chosen:
            period = str(o.validity_days)+' days' if o.validity_days else 'validity NOT clearly stated'
            output.append(f'• {o.category.upper()} | {o.description} | BDT {o.price_bdt} | '
                          f'{period} | USSD {o.ussd}\n  Exact table row: {o.evidence}')
        output.append(f'Official source: {page.url}\nFetched UTC: {page.fetched_at_utc}')
        output.append('Special offers/availability may depend on your SIM. Confirm activation and eligibility with Teletalk.')
        return KnowledgeAnswer('live_public_listing',label,'\n'.join(output),
                               (page.url,),tuple(o.ussd for o in chosen),provider='official-website')
    except (LiveSourceError,requests.RequestException,ValueError,KeyError,FileNotFoundError) as exc:
        return KnowledgeAnswer('unavailable',label,
            f'Could not verify a current official page ({type(exc).__name__}). '
            'No cached offer price is presented as current. Check the official Teletalk website directly.',
            (last_source,),provider='official-unavailable')


def _audit_read(page: LivePage, *,kind:str,count:int,skipped=None,offers=None):
    """Append content evidence metadata only. Never log customer query, phone, or NID."""
    AUDIT_DIR.mkdir(parents=True,exist_ok=True)
    record = {'kind':kind,'source_url':page.url,'fetched_at_utc':page.fetched_at_utc,
              'source_sha256':page.sha256,'extracted_count':count,'skipped':skipped or [],
              'observed_offers':[asdict(o) for o in (offers or [])[:150]],
              'note':'Official HTML as fetched, NOT proof of live activation or subscriber eligibility'}
    with (AUDIT_DIR/'live_reads.jsonl').open('a',encoding='utf-8') as out:
        out.write(json.dumps(record,ensure_ascii=False)+'\n')
