from datetime import date,timedelta
from collections import Counter
import pytest
from src.teletalk_kb import (load_offers,load_info,load_conflicts,official_url,MAX_AGE_DAYS,Catalog,Offer,InfoPage,CONFLICTS)

TODAY=date(2026,11,2)

def test_offers_approved():
    c=load_offers(today=TODAY)
    assert len(c.offers)==73
    assert len({x.id for x in c.offers})==73
    assert all(x.availability=='publicly_listed_unconfirmed' for x in c.offers)

def test_all_offer_categories():
    counts=Counter(x.category for x in load_offers(today=TODAY).offers)
    assert set(['internet','combo','minute','sms','unlimited_weekend']) <= set(counts)

def test_special_subscriber_not_verified():
    restricted=[o for o in load_offers(today=TODAY).offers if o.sim_type!='public']
    assert restricted and all(o.eligibility_status=='requires_external_verification' for o in restricted)

def test_direct_sources():
    for row in load_offers(today=TODAY).offers:
        assert official_url(row.source_url)==row.source_url

def test_info_pages():
    x=load_info(today=TODAY)
    assert len(x)==18
    assert all(x.source_url.startswith('https://') for x in x)

def test_price_conflict_quarantined():
    blocked={r['code'] for r in load_conflicts()}
    assert '*111*8#' in blocked
    assert not any(o.ussd in blocked for o in load_offers(today=TODAY).offers)

def test_future_refusal():
    with pytest.raises(ValueError,match='Future'):
        load_offers(today=TODAY-timedelta(days=1))

def test_expiration():
    with pytest.raises(ValueError,match='too old'):
        load_offers(today=TODAY+timedelta(days=MAX_AGE_DAYS+1))

def test_source_host_allowlist():
    for bad in ['http://www.teletalk.com.bd/a','https://teletalk.com.bd.evil.test/a','https://user@teletalk.com.bd/a',
                'https://teletalk.com.bd:443/a','https://teletalk.com.bd/a?foo=x']:
        with pytest.raises(ValueError):official_url(bad)

def test_code_validation():
    x=load_offers(today=TODAY).offers[0].model_dump()
    x['ussd']='*111*bad#'
    with pytest.raises(Exception):Offer.model_validate(x)

def test_no_manual_price_in_forbidden_genz_ussd():
    g=[o for o in load_offers(today=TODAY).offers if o.sim_type=='gen_z']
    assert len(g)==4 and all(o.ussd is None for o in g)
