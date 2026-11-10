"""Source-locked Teletalk public knowledge. No customer data or synthetic offers."""
from __future__ import annotations
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Literal
from pathlib import Path
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'knowledge' / 'stage7_full'
OFFERS = STORE / 'approved_offers.json'
INFO = STORE / 'approved_info.json'
CONFLICTS = STORE / 'quarantined_conflicts.json'
REVIEWED = STORE / 'reviewed_live_pages.json'
MAX_AGE_DAYS = 14


def official_url(value: str) -> str:
    """Reject non-Teletalk URLs, credentials, unexpected ports, query based sources."""
    if not isinstance(value, str) or len(value) > 400:
        raise ValueError('Invalid official URL.')
    p = urlsplit(value)
    if p.scheme != 'https' or p.hostname not in ('teletalk.com.bd', 'www.teletalk.com.bd') or p.username or p.password or p.port or p.query or p.fragment or not p.path.startswith('/'):
        raise ValueError('Only official Teletalk HTTPS pages are accepted.')
    return value


class Offer(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    id: str = Field(min_length=7, max_length=130)
    title: str = Field(min_length=3, max_length=180)
    category: str = Field(pattern=r'^(internet|combo|minute|sms|unlimited_weekend)$')
    price_bdt: int = Field(ge=0, le=100_000)
    validity_days: int | None = Field(default=None, ge=1, le=366)
    data_mb: int | None = Field(default=None, ge=1, le=500_000)
    minutes: int | None = Field(default=None, ge=1, le=50000)
    sms: int | None = Field(default=None, ge=1, le=50000)
    ussd: str | None = None
    sim_type: str = Field(min_length=2, max_length=60)
    source_url: str
    checked_on: date
    eligibility_status: str = Field(pattern=r'^(public_listing|requires_external_verification)$')
    eligibility_conditions: list[str]
    restrictions: list[str]
    activation: str
    availability: str = Field(pattern=r'^publicly_listed_unconfirmed$')

    @field_validator('checked_on', mode='before')
    @classmethod
    def checked_date(cls, value):
        if isinstance(value,str) and len(value)==10:
            return date.fromisoformat(value)
        return value

    @field_validator('source_url')
    @classmethod
    def check_source(cls, value):
        return official_url(value)

    @field_validator('ussd')
    @classmethod
    def check_code(cls, value):
        import re
        if value is not None and re.fullmatch(r'\*[0-9*]+#', value) is None:
            raise ValueError('USSD malformed.')
        return value


class InfoPage(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    id: str
    title: str
    category: str
    source_url: str
    checked_on: date
    body: str = Field(min_length=20, max_length=10000)

    @field_validator('checked_on', mode='before')
    @classmethod
    def checked_date(cls, value):
        if isinstance(value,str) and len(value)==10:
            return date.fromisoformat(value)
        return value

    @field_validator('source_url')
    @classmethod
    def check_source(cls, value):
        return official_url(value)


class Catalog(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[2]
    snapshot_compiled_on: date
    notice: str
    offers: list[Offer]
    @model_validator(mode='after')
    def unique(self):
        ids=[r.id for r in self.offers]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate offer id')
        return self


def _fresh(d: date, now: date | None = None, *, allow_stale=False):
    age=((now or datetime.now(ZoneInfo("Asia/Dhaka")).date())-d).days
    if age < 0: raise ValueError('Future dated source refused')
    if age > MAX_AGE_DAYS and not allow_stale:
        raise ValueError(f'Official snapshot too old ({age} days); re-review required. Do not serve prices.')


def load_offers(*, path: Path=OFFERS, today: date | None=None, allow_stale=False) -> Catalog:
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    cat=Catalog.model_validate(raw)
    _fresh(cat.snapshot_compiled_on, today, allow_stale=allow_stale)
    for row in cat.offers:
        _fresh(row.checked_on, today, allow_stale=allow_stale)
    conflicts=load_conflicts()
    blocked_codes={x.get('code') for x in conflicts if x.get('code')}
    if any(o.ussd in blocked_codes for o in cat.offers):
        raise ValueError('Conflicting offer code present in approved catalog. Refusing all prices.')
    return cat


def load_conflicts(path:Path=CONFLICTS) -> list[dict]:
    return json.loads(Path(path).read_text(encoding='utf-8')).get('conflicts',[])


def load_info(*, path:Path=INFO, today:date|None=None, allow_stale=False) -> list[InfoPage]:
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    _fresh(date.fromisoformat(obj['snapshot_compiled_on']), today, allow_stale=allow_stale)
    result=[InfoPage.model_validate(p) for p in obj['pages']]
    for row in result: _fresh(row.checked_on, today, allow_stale=allow_stale)
    if REVIEWED.exists() and path == INFO:
        raw=json.loads(REVIEWED.read_text(encoding='utf-8'))
        for p in raw.get('pages',[]):
            row=InfoPage.model_validate(p)
            _fresh(row.checked_on, today, allow_stale=allow_stale)
            result.append(row)
    ids=[x.id for x in result]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate reviewed page IDs.')
    return result


def catalog_fingerprint() -> str:
    inputs=[OFFERS, INFO, CONFLICTS]
    if REVIEWED.exists(): inputs.append(REVIEWED)
    raw=b''.join(hashlib.sha256(p.read_bytes()).digest() for p in inputs)
    return hashlib.sha256(raw).hexdigest()


def document_text(row: Offer|InfoPage) -> str:
    if isinstance(row,Offer):
        return ('Teletalk public offer. '+row.title+'; category '+row.category+'; '
                +f'price BDT {row.price_bdt}; validity {row.validity_days} days; '
                +f'data {row.data_mb} MB; minutes {row.minutes}; SMS {row.sms}; '
                +f'SIM {row.sim_type}; conditions '+ '; '.join(row.eligibility_conditions)
                +'; restrictions '+ '; '.join(row.restrictions)+'; '
                +'activation '+row.activation+'; source '+row.source_url)
    return f'Teletalk official public information: {row.title}. {row.body} Source: {row.source_url}'
