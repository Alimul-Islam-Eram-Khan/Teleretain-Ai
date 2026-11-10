"""No production network or subscriber records. Evidence mirrors supplied public listing logs."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import pytest
from pydantic import ValidationError
from src.teletalk_live import LiveOffer, LivePage, LiveSourceError
from src.teletalk_answer import KnowledgeAnswer
from src.retention_recommender import (
    RecommendationRequest as Req, recommend, from_question, metrics, present,
)

URL='https://www.teletalk.com.bd/internet-offers'

def seed():
    stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    page=LivePage(URL,'<html><main>Official package details for research tests</main></html>',
                  'Official package details for research tests','Offers',stamp,hashlib.sha256(b'html').hexdigest())
    offers=[
        LiveOffer('internet','100 MB',11,7,'*111*501#',URL,stamp,'100 MB | 7 Days | TK 11 | *111*501# | Yes'),
        LiveOffer('internet','500 MB',18,7,'*111*717#',URL,stamp,'500 MB | 7 Days | TK 18 | *111*717# | Yes'),
        LiveOffer('internet','1 GB',22,7,'*111*534#',URL,stamp,'1 GB | 7 Days | TK 22 | *111*534# | Yes'),
        LiveOffer('internet','2GB+ 102 MB Bonus',38,7,'*111*736#',URL,stamp,'2GB+ 102 MB Bonus | 7 Days | TK 38 | *111*736# | Yes'),
        LiveOffer('combo','1 GB+25 minute+10 SMS',47,30,'*111*118#',URL,stamp,'30 days | Tk 47 | *111*118# | Yes'),
        LiveOffer('combo','1 GB+55 minute+50 SMS',66,30,'*111*121#',URL,stamp,'30 days | Tk 66 | *111*121# | Yes'),
        LiveOffer('minute','Minute_Bundle_23 Min',15,3,'*111*7*1#',URL,stamp,'Minute_Bundle_23 Min | 3 Days | Tk 15 | *111*7*1# | Yes'),
        LiveOffer('sms','SMS_Bundle_Pack50',7,3,'*111*6#',URL,stamp,'SMS_Bundle_Pack50 | 3 Days | Tk 7 | *111*6# | Yes'),
    ]
    return page,offers,[]

def load(_):return seed()

def test_internet_budget_validity():
    r=recommend(Req(category='internet',budget_bdt=22,exact_validity_days=7),fetcher=object(),offer_loader=load)
    assert r.status=='recommended'
    assert [o.price_bdt for o in r.items]==[11,18,22]
    assert r.items[0].source_url==URL and r.source_sha256

def test_combo_requirements():
    r=recommend(Req(category='combo',budget_bdt=100,min_minutes=50,min_sms=50,min_data_mb=1024),fetcher=object(),offer_loader=load)
    assert r.status=='recommended' and len(r.items)==1 and r.items[0].price_bdt==66

def test_zero_assumptions_on_bonus_data():
    assert metrics('2GB+ 102 MB Bonus','internet')[0]==2048
    r=recommend(Req(category='internet',min_data_mb=2100),fetcher=object(),offer_loader=load)
    assert r.status=='no_match'

def test_missing_explicit_sms_counts_not_guessed():
    r=recommend(Req(category='sms',min_sms=50),fetcher=object(),offer_loader=load)
    assert r.status=='no_match'

def test_minutes_exact():
    r=recommend(Req(category='minute',min_minutes=23,exact_validity_days=3),fetcher=object(),offer_loader=load)
    assert len(r.items)==1 and r.items[0].minutes_explicit==23

def test_sort_most_data():
    r=recommend(Req(category='internet',sort_by='most_data'),fetcher=object(),offer_loader=load)
    assert r.items[0].data_mb_conservative==2048

def test_sort_value():
    r=recommend(Req(category='internet',sort_by='data_per_taka'),fetcher=object(),offer_loader=load)
    assert r.items[0].ussd=='*111*736#'

def test_longest_validity():
    r=recommend(Req(category='combo',sort_by='longest_validity'),fetcher=object(),offer_loader=load)
    assert len(r.items)==2

def test_no_match_is_not_claim_that_no_offer_exists():
    r=recommend(Req(category='internet',budget_bdt=1),fetcher=object(),offer_loader=load)
    assert r.status=='no_match' and 'other Teletalk offers' in r.message

def test_conflict_quarantine():
    p,os,x=seed()
    os.append(replace(os[0],price_bdt=99, description='100 MB new price'))
    r=recommend(Req(category='internet',budget_bdt=25),fetcher=object(),offer_loader=lambda _: (p,os,x))
    assert r.excluded_conflicts==1
    assert '*111*501#' not in [v.ussd for v in r.items]

def test_no_stale_data_used():
    p,os,x=seed()
    p=replace(p,fetched_at_utc=(datetime.now(timezone.utc)-timedelta(minutes=10)).isoformat())
    r=recommend(Req(category='internet'),fetcher=object(),offer_loader=lambda _: (p,os,x))
    assert r.status=='unavailable' and not r.items and 'No historical price' in r.message

def test_source_domain_spoofing_rejected():
    p,os,x=seed()
    p=replace(p,url='https://teletalk.com.bd.evil.example/internet-offers')
    r=recommend(Req(category='internet'),fetcher=object(),offer_loader=lambda _: (p,os,x))
    assert r.status=='unavailable' and not r.items

def test_hash_provenance_missing():
    p,os,x=seed()
    p=replace(p,sha256='garbage')
    assert recommend(Req(category='internet'),fetcher=object(),offer_loader=lambda _: (p,os,x)).status=='unavailable'

def test_inconsistent_timestamp():
    p,os,x=seed()
    os[0]=replace(os[0],fetched_at_utc='2001-01-01T00:00:00+00:00')
    assert recommend(Req(category='internet'),fetcher=object(),offer_loader=lambda _: (p,os,x)).status=='unavailable'

def test_source_evidence_required():
    p,os,x=seed()
    os[0]=replace(os[0],evidence='')
    assert recommend(Req(category='internet'),fetcher=object(),offer_loader=lambda _: (p,os,x)).status=='unavailable'

def test_network_fails_closed():
    def fail(_): raise LiveSourceError('Site unavailable')
    r=recommend(Req(category='internet'),fetcher=object(),offer_loader=fail)
    assert r.status=='unavailable' and not r.items and 'BDT 11' not in present(r)

def test_special_sim_requires_external_eligibility():
    calls=[]
    def info(question, *, fetcher):
        calls.append(question)
        return KnowledgeAnswer('live_source_information','Official',
                'Only selected customers may participate; operator-side check is required.',
                ('https://www.teletalk.com.bd/en/voice/prepaid/closed-sim-offer-promotion',))
    r=recommend(Req(sim_type='closed_sim'),fetcher=object(),
                offer_loader=lambda _:pytest.fail('Must not price special SIM'),info_lookup=info)
    assert r.status=='eligibility_verification_required' and not r.items
    assert calls == ['closed sim eligibility conditions']

def test_special_sim_missing_evidence_fails():
    def bad(question, *, fetcher):
        return KnowledgeAnswer('unavailable','Official','Cannot verify')
    r=recommend(Req(sim_type='gen_z'),fetcher=object(),offer_loader=load,info_lookup=bad)
    assert r.status=='unavailable'

def test_question_banglish_parsing():
    r=from_question('100 takar moddhe 7 diner internet pack',fetcher=object(),offer_loader=load)
    assert r.status=='recommended' and len(r.items)==4

def test_question_bangla_parsing():
    r=from_question('১০০ টাকার মধ্যে ৭ দিনের ইন্টারনেট প্যাকেজ',fetcher=object(),offer_loader=load)
    assert r.status=='recommended' and all(o.validity_days==7 for o in r.items)

def test_question_ambiguous():
    r=from_question('amar kon package valo?',fetcher=object(),offer_loader=load)
    assert r.status=='needs_clarification'

def test_question_general_service_delegated():
    r=from_question('customer care eligibility information',fetcher=object(),offer_loader=load)
    assert r.status=='needs_clarification'

def test_wrong_type_extra_key():
    with pytest.raises(ValidationError):Req(category='internet',credit_card='123')

def test_bool_not_int():
    with pytest.raises(ValidationError):Req(category='internet',budget_bdt=True)

def test_budget_range():
    with pytest.raises(ValidationError):Req(category='internet',budget_bdt=-1)

def test_inconsistent_days():
    with pytest.raises(ValidationError):Req(category='internet',exact_validity_days=3,min_validity_days=7)

def test_impossible_category_mix():
    with pytest.raises(ValidationError):Req(category='internet',min_minutes=25)

def test_value_ranking_requires_data_category():
    with pytest.raises(ValidationError):Req(category='sms',sort_by='data_per_taka')

def test_limit_enforced():
    with pytest.raises(ValidationError):Req(category='minute',limit=100)

def test_sim_type_strict():
    with pytest.raises(ValidationError):Req(category='internet',sim_type='super_vip')

def test_output_deterministic_and_cited():
    req=Req(category='internet',language='bangla',limit=2)
    first=recommend(req,fetcher=object(),offer_loader=load)
    second=recommend(req,fetcher=object(),offer_loader=load)
    assert first.to_dict()['items']==second.to_dict()['items']
    assert 'অফিসিয়াল' in present(first,'bangla')
    assert 'Source:' in present(first,'bangla')
    assert first.citations==(URL,)

def test_limit_does_not_change_matched_total():
    r=recommend(Req(category='internet',limit=1),fetcher=object(),offer_loader=load)
    assert r.matched_total==4 and len(r.items)==1

def test_unverified_activation_never_claimed():
    r=recommend(Req(category='combo'),fetcher=object(),offer_loader=load)
    assert all('NOT VERIFIED' in i.eligibility_status for i in r.items)
    assert 'not a personal account' in r.disclaimer
