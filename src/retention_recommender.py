"""Stage 8: deterministic, official-live, eligibility-aware PUBLIC-OFFER matching.

This is offer-fit assistance, NOT real Teletalk churn personalization.
Never interpret public listings as account activation/eligibility verification.
No generated prices, no LLM, no private customer data, no stale fallback.
Compatible with Stage 7 v3.4 official fetcher and parser.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Literal
import math
import re
import requests

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.teletalk_answer import parse_question
from src.teletalk_live import (
    LiveOffer, LivePage, LiveSourceError, OfficialFetcher, answer_live,
    fetch_offers_fresh, official,
)

Categories = Literal['internet', 'combo', 'minute', 'sms', 'unlimited_weekend']
SortOrder = Literal['lowest_price', 'most_data', 'longest_validity', 'data_per_taka']
Language = Literal['english', 'bangla', 'banglish']
SimType = Literal['public', 'gen_z', 'professional', 'closed_sim', 'oporajita',
                  'mayer_hasi', 'agami', 'bornomala']

SPECIAL_QUESTIONS = {
    'gen_z': 'Gen-Z eligibility conditions',
    'professional': 'professional SIM eligibility conditions',
    'closed_sim': 'closed sim eligibility conditions',
    'oporajita': 'Oporajita eligibility conditions',
    'mayer_hasi': 'Mayer Hasi eligibility conditions',
    'agami': 'Agami eligibility conditions',
    'bornomala': 'Bornomala eligibility conditions',
}
PUBLIC_DISCLAIMER = (
    'These are newly fetched PUBLIC Teletalk listings, not a personal account eligibility, '
    'successful activation, or guaranteed retention benefit. Verify conditions and activation '
    'with Teletalk. Iranian/IBM churn models were not trained on Teletalk customers.'
)

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    category: Categories | None = None
    budget_bdt: int | None = Field(default=None, ge=1, le=100_000)
    exact_validity_days: int | None = Field(default=None, ge=1, le=366)
    min_validity_days: int | None = Field(default=None, ge=1, le=366)
    min_data_mb: int | None = Field(default=None, ge=1, le=500_000)
    min_minutes: int | None = Field(default=None, ge=1, le=50_000)
    min_sms: int | None = Field(default=None, ge=1, le=50_000)
    sim_type: SimType = 'public'
    sort_by: SortOrder = 'lowest_price'
    language: Language = 'english'
    limit: int = Field(default=5, ge=1, le=10)

    @model_validator(mode='after')
    def verify_consistency(self):
        if self.exact_validity_days and self.min_validity_days and self.exact_validity_days < self.min_validity_days:
            raise ValueError('Exact validity is shorter than the requested minimum.')
        if self.category == 'internet' and (self.min_minutes or self.min_sms):
            raise ValueError('For both data and minutes/SMS, choose combo category.')
        if self.category == 'minute' and (self.min_data_mb or self.min_sms):
            raise ValueError('Minute-only bundles do not guarantee data or SMS.')
        if self.category == 'sms' and (self.min_data_mb or self.min_minutes):
            raise ValueError('SMS-only bundles do not guarantee data or minutes.')
        if self.category is None and self.sim_type == 'public':
            raise ValueError('Specify category (internet/combo/minute/sms/unlimited_weekend).')
        if self.sort_by == 'data_per_taka' and self.category not in ('internet', 'combo'):
            raise ValueError('Data-per-taka ranking requires internet or combo category.')
        return self


@dataclass(frozen=True)
class RecommendationItem:
    rank: int
    category: str
    description: str
    price_bdt: int
    validity_days: int | None
    ussd: str
    data_mb_conservative: int | None
    minutes_explicit: int | None
    sms_explicit: int | None
    reason: str
    eligibility_status: str
    source_url: str
    fetched_at_utc: str
    exact_source_row: str


@dataclass(frozen=True)
class RecommendationResult:
    status: str
    message: str
    items: tuple[RecommendationItem, ...] = ()
    source_url: str | None = None
    fetched_at_utc: str | None = None
    source_sha256: str | None = None
    matched_total: int = 0
    excluded_conflicts: int = 0
    citations: tuple[str, ...] = ()
    disclaimer: str = PUBLIC_DISCLAIMER

    def to_dict(self):
        return asdict(self)


def metrics(description: str, category: str) -> tuple[int | None, int | None, int | None]:
    """Conservative amount extraction: count only explicit BASE sizes, not bonus/promises.

    '2GB+102MB Bonus' counts 2GB (not bonus); unknown amounts stay None.
    These metrics are not measurements of subscriber usage.
    """
    text = description.casefold().replace(',', '')
    data = re.search(r'(\d+(?:\.\d+)?)\s*(gb|mb)\b', text)
    base_mb = None
    if data:
        # A description beginning with 'Bonus' cannot establish base entitlement.
        preamble = text[:data.start()]
        if not re.search(r'bonus\s*$', preamble):
            base_mb = math.floor(float(data.group(1)) * (1024 if data.group(2) == 'gb' else 1))
    minutes = re.search(r'(\d+)\s*(?:minutes?|min)\b', text)
    sms = re.search(r'(\d+)\s*sms\b', text)
    return (
        base_mb if base_mb and category in ('internet', 'combo', 'unlimited_weekend') else None,
        int(minutes.group(1)) if minutes and category in ('minute', 'combo') else None,
        int(sms.group(1)) if sms and category in ('sms', 'combo') else None,
    )


def _check_page(page: LivePage, offers: list[LiveOffer]):
    url = official(page.url)
    if url != page.url or not re.fullmatch(r'[0-9a-f]{64}', page.sha256):
        raise LiveSourceError('Unverifiable source URL or page hash')
    try:
        stamp = datetime.fromisoformat(page.fetched_at_utc)
        if stamp.tzinfo is None:
            raise ValueError('Naive timestamp')
        age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
        if not -60 <= age <= 300:
            raise ValueError('Not a fresh official fetch')
    except (ValueError, TypeError) as exc:
        raise LiveSourceError('Offer source freshness cannot be established') from exc
    if len(offers) > 500:
        raise LiveSourceError('Unusually large page; manual source review needed')
    for offer in offers:
        if not isinstance(offer, LiveOffer) or official(offer.source_url) != page.url or offer.fetched_at_utc != page.fetched_at_utc:
            raise LiveSourceError('Offer provenance does not match the fetched source')
        if not re.fullmatch(r'\*[0-9*]+#', offer.ussd) or not 0 < offer.price_bdt <= 100_000:
            raise LiveSourceError('Invalid official offer fields')
        if not offer.evidence or len(offer.evidence) > 500:
            raise LiveSourceError('Missing original source-table evidence')


def _matches(offer: LiveOffer, req: RecommendationRequest, met: tuple):
    data, minutes, sms = met
    if req.category and offer.category != req.category:
        return False
    if req.budget_bdt is not None and offer.price_bdt > req.budget_bdt:
        return False
    if req.exact_validity_days is not None and offer.validity_days != req.exact_validity_days:
        return False
    if req.min_validity_days is not None and (offer.validity_days is None or offer.validity_days < req.min_validity_days):
        return False
    if req.min_data_mb is not None and (data is None or data < req.min_data_mb):
        return False
    if req.min_minutes is not None and (minutes is None or minutes < req.min_minutes):
        return False
    if req.min_sms is not None and (sms is None or sms < req.min_sms):
        return False
    return True


def _rank(row, order):
    o, (data, minutes, sms) = row
    # Unstated amounts cannot win a quality-based ranking.
    if order == 'most_data':
        return (data is None, -(data or 0), o.price_bdt, o.ussd)
    if order == 'longest_validity':
        return (o.validity_days is None, -(o.validity_days or 0), o.price_bdt, o.ussd)
    if order == 'data_per_taka':
        return (data is None, -(data or 0) / o.price_bdt, o.price_bdt, o.ussd)
    return (o.price_bdt, -(o.validity_days or 0), -(data or 0), o.ussd)


def _reason(req, offer, met):
    details = [f'BDT {offer.price_bdt}']
    if req.budget_bdt is not None: details.append('within stated budget')
    if offer.validity_days is not None: details.append(f'{offer.validity_days} days as listed')
    if req.min_data_mb is not None: details.append('explicit base data meets requested minimum')
    if req.min_minutes is not None: details.append('explicit minutes meet requested minimum')
    if req.min_sms is not None: details.append('explicit SMS count meets requested minimum')
    return '; '.join(details) + '. Public offer-fit match only, not churn reduction evidence.'


def recommend(request: RecommendationRequest, *, fetcher=None, offer_loader=fetch_offers_fresh,
              info_lookup=answer_live) -> RecommendationResult:
    """A fresh fetch for EVERY request. A failed fetch NEVER uses stored price as current.

    Inject loaders for tests only. Live use takes Stage 7's OfficialFetcher/Schannel.
    """
    if not isinstance(request, RecommendationRequest):
        raise TypeError('Validated RecommendationRequest required')
    source = 'https://www.teletalk.com.bd/internet-offers'
    try:
        client = fetcher if fetcher is not None else OfficialFetcher()
        if request.sim_type != 'public':
            info = info_lookup(SPECIAL_QUESTIONS[request.sim_type], fetcher=client)
            if info.mode != 'live_source_information' or not info.citations:
                return RecommendationResult('unavailable', 'Current official eligibility conditions could not be verified. No special price claimed.')
            for link in info.citations: official(link)
            return RecommendationResult(
                'eligibility_verification_required',
                'Public SIM/package eligibility information only; operator-side account verification is required.\n' + info.message,
                source_url=info.citations[0], citations=tuple(info.citations),
            )
        page, offers, exclusions = offer_loader(client)
        _check_page(page, offers)
        source = page.url
        # Defense-in-depth: quarantine any conflicting USSD even if the Stage 7 parser changes.
        seen: dict[str, set] = {}
        for o in offers:
            seen.setdefault(o.ussd, set()).add((o.category, o.description, o.price_bdt, o.validity_days))
        conflicts = {code for code, values in seen.items() if len(values) != 1}
        rows = [(o, metrics(o.description, o.category)) for o in offers if o.ussd not in conflicts]
        rows = [row for row in rows if _matches(row[0], request, row[1])]
        rows.sort(key=lambda row: _rank(row, request.sort_by))
        items = tuple(
            RecommendationItem(
                rank=i, category=o.category, description=o.description, price_bdt=o.price_bdt,
                validity_days=o.validity_days, ussd=o.ussd,
                data_mb_conservative=met[0], minutes_explicit=met[1], sms_explicit=met[2],
                reason=_reason(request, o, met),
                eligibility_status='PUBLIC LISTING ONLY — SIM/ACTIVATION NOT VERIFIED',
                source_url=page.url, fetched_at_utc=page.fetched_at_utc, exact_source_row=o.evidence,
            ) for i, (o, met) in enumerate(rows[:request.limit], 1)
        )
        if not items:
            return RecommendationResult(
                'no_match', 'No matching *explicitly listed* package was found on the freshly fetched official page. '
                'This does not mean other Teletalk offers do not exist.',
                source_url=page.url, fetched_at_utc=page.fetched_at_utc, source_sha256=page.sha256,
                excluded_conflicts=len(conflicts), citations=(page.url,),
            )
        return RecommendationResult(
            'recommended', f'{len(items)} source-matched public listing(s), ordered by {request.sort_by}. '
            'This is not a customer-specific retention intervention.', items=items,
            source_url=page.url, fetched_at_utc=page.fetched_at_utc,
            source_sha256=page.sha256, matched_total=len(rows), excluded_conflicts=len(conflicts),
            citations=(page.url,),
        )
    except (LiveSourceError, ValueError, OSError, TypeError, KeyError, requests.RequestException) as exc:
        return RecommendationResult(
            'unavailable', 'A fresh official listing could not be safely verified '
            f'({type(exc).__name__}). No historical price has been presented as current.',
        )


def from_question(question: str, *, fetcher=None, offer_loader=fetch_offers_fresh,
                  info_lookup=answer_live, language: Language='banglish', limit: int=5,
                  sort_by: SortOrder='lowest_price') -> RecommendationResult:
    """Bangla/Banglish parser is Stage 7's tested deterministic parser."""
    try:
        p = parse_question(question)
        if p.get('info_request') and not p.get('sim_type'):
            return RecommendationResult('needs_clarification',
                'For general service/FAQ questions use Stage 7 source retrieval. For packages specify a type and budget.')
        if p['category'] is None and not p['sim_type']:
            return RecommendationResult('needs_clarification',
                'Specify Internet, Combo, Minute, SMS or Unlimited/Weekend package category.')
        return recommend(RecommendationRequest(
            category=p['category'], budget_bdt=p['budget_bdt'],
            min_validity_days=p['min_validity_days'] if p['exact_validity_days'] is None else None,
            exact_validity_days=p['exact_validity_days'],
            min_data_mb=p['min_data_mb'], min_minutes=p['min_minutes'], min_sms=p['min_sms'],
            sim_type=p['sim_type'] or 'public', language=language, limit=limit, sort_by=sort_by,
        ), fetcher=fetcher, offer_loader=offer_loader, info_lookup=info_lookup)
    except (ValueError, TypeError) as exc:
        return RecommendationResult('needs_clarification', f'Please clarify the request ({type(exc).__name__}); no price asserted.')


def present(result: RecommendationResult, language: Language='banglish') -> str:
    """Deterministic output; grounded numbers come solely from structured live offers."""
    if not isinstance(result, RecommendationResult): raise TypeError('Invalid result')
    intro = {
        'english': 'Source-grounded public offer matches:',
        'bangla': 'অফিসিয়াল ওয়েবসাইট থেকে যাচাইকৃত পাবলিক অফার:',
        'banglish': 'Official website theke verified public offer:',
    }[language]
    lines = [intro, f'Status: {result.status}', result.message]
    for item in result.items:
        validity = f'{item.validity_days} days' if item.validity_days else 'Validity not confirmed in days'
        lines.extend([
            f'{item.rank}. {item.category.upper()}: {item.description} | BDT {item.price_bdt} | {validity} | USSD {item.ussd}',
            f'   Why: {item.reason}',
            f'   Evidence: {item.exact_source_row}',
            f'   Source: {item.source_url} | Fetched UTC: {item.fetched_at_utc}',
        ])
    if result.source_url and not result.items:
        lines.append(f'Source: {result.source_url} | Fetched UTC: {result.fetched_at_utc or "not established"}')
    lines.append(result.disclaimer)
    return '\n'.join(lines)
