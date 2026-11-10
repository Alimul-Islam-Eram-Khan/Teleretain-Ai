from datetime import date,timedelta
import pytest
from src.teletalk_answer import (parse_question,ask_teletalk,answer_offers,answer_information)
TODAY=date(2026,11,2)

@pytest.mark.parametrize('query',[
    '100 takar moddhe 7 diner internet package chai',
    'amar 7 diner 100 tk er moddhe internet pack chai',
    '১০০ টাকার মধ্যে ৭ দিনের ইন্টারনেট প্যাকেজ',
    'internet 7 days under 100 taka',
])
def test_bilingual_budgets(query):
    p=parse_question(query)
    assert p['budget_bdt']==100 and p['min_validity_days']==7

def test_never_confuse_days_with_taka():
    p=parse_question('7 diner 100 taka er moddhe net pack')
    assert p['budget_bdt']==100

def test_basic_price_filter():
    r=answer_offers('internet 30 days under 100 taka',today=TODAY)
    assert r.mode=='grounded' and r.citations
    assert 'BDT 56' in r.message or 'BDT 88' in r.message or 'BDT 23' in r.message
    assert 'public' in r.message.lower()

def test_combo_filter():
    r=answer_offers('combo under 120 taka 30 days',today=TODAY)
    assert r.mode=='grounded' and all('teletalk.com.bd' in x for x in r.citations)

def test_unsupported_budget_refused():
    r=answer_offers('internet 50GB 30 days under 20 taka',today=TODAY)
    assert r.mode=='no_match' and not r.offer_ids

def test_eligible_never_automatically_claimed():
    r=answer_offers('professional 10GB package under 500 taka',today=TODAY)
    assert r.mode=='grounded'
    assert 'ELIGIBILITY NOT VERIFIED' in r.message

def test_generic_does_not_surface_special_sim():
    r=answer_offers('internet 7 days under 25 taka',today=TODAY)
    assert r.mode=='grounded'
    assert not any('Gen-Z' in line for line in r.message.splitlines())

def test_exact_profile_answer():
    r=ask_teletalk('Oporajita eligibility conditions',today=TODAY)
    assert r.mode=='grounded' and 'women' in r.message.lower() and r.citations

def test_closed_sim_is_not_guaranteed():
    r=ask_teletalk('closed sim eligibility conditions',today=TODAY)
    assert r.mode=='grounded' and 'does not check subscriber eligibility' in r.message

def test_no_old_price_claims():
    r=answer_offers('internet under 100 taka',today=TODAY+timedelta(days=20))
    assert r.mode=='unavailable' and 'BDT 56' not in r.message

def test_bad_question():
    assert answer_offers('',today=TODAY).mode=='unavailable'

def test_monthly_7_day_no_confusion():
    r=answer_offers('minimum 30 days 2GB internet under 100 taka',today=TODAY)
    assert r.mode=='grounded' and '2 GB' in r.message

def test_query_number_derived():
    p=parse_question('I need 10GB internet 30 days under 300 taka')
    assert p['min_data_mb']==10240 and p['min_validity_days']==30 and p['budget_bdt']==300

def test_exact_duration_is_not_minimum_duration():
    p=parse_question('7 days internet under 100 taka')
    assert p['exact_validity_days']==7
    r=answer_offers('7 days internet under 100 taka',today=TODAY)
    assert r.mode=='grounded' and '30 days' not in r.message

def test_minimum_duration_includes_longer_validity():
    p=parse_question('minimum 7 days internet under 100 taka')
    assert p['exact_validity_days'] is None
    assert p['min_validity_days']==7

def test_specific_ussd_code_match():
    r=ask_teletalk('What is the price for *111*534#?',today=TODAY)
    assert r.mode=='grounded' and 'BDT 22' in r.message and '*111*534#' in r.message

def test_conflicting_specific_ussd_code_quarantined():
    r=ask_teletalk('Tell me about *111*8#',today=TODAY)
    assert r.mode=='unavailable' and 'conflicting' in r.message
