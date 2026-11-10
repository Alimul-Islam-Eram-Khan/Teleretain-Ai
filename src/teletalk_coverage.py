"""Coverage and exception audit; never claim 100% of a changing website."""
from __future__ import annotations
from collections import Counter
from datetime import date
from pathlib import Path
import json
from src.teletalk_kb import STORE,OFFERS,INFO,REVIEWED,load_offers,load_info,load_conflicts
from src.teletalk_discovery import REPORT,QUEUE

SCOPE={
 'internet':'regular internet and bonus data',
 'combo':'data, voice and SMS bundles',
 'minute':'minute bundles',
 'sms':'SMS bundles',
 'unlimited_weekend':'unlimited-validity, uninterrupted, weekend',
 'sim_eligibility':'Agami, Bornomala, Gen-Z, Oporajita, Mayer Hasi, professional, closed SIM',
 'prepaid_profiles':'other prepaid variants and migration policies',
 'postpaid':'postpaid products and billing',
 'corporate':'corporate packs and tariff',
 'roaming':'international roaming and ISD',
 'vas':'VAS, emergency balance, balance transfer, call management',
 'faq':'FAQs including 4G and online service support',
 'customer_care':'care centres and complaints',
 'devices':'modems, Mi-Fi, eSIM and routers',
 'teleplan':'dynamic customized bundle builder',
 'promotions':'limited campaign, recharge, rate-cutter and scratch bonuses',
 'network':'coverage and connectivity',
 'other':'other official pages present in Teletalk sitemap',
}

def coverage(*,today:date|None=None):
    o=load_offers(today=today)
    pages=load_info(today=today)
    d=json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else None
    counter=Counter(x.category for x in o.offers)
    result={
      'source_status':'dated official web snapshot; not proof all live pages were captured',
      'approved_offers':len(o.offers),'reviewed_information_entries':len(pages),
      'offer_categories':dict(counter),'source_urls':len({x.source_url for x in list(o.offers)+pages}),
      'quarantined_conflicts':load_conflicts(),
      'discovery':d,'review_queue_count':len(list(QUEUE.glob('*.json'))) if QUEUE.exists() else 0,
      'scope':[],
    }
    for key,description in SCOPE.items():
        count=counter.get(key,0)
        references=sum(1 for p in pages if key in p.category or key in p.id or key.replace('_',' ') in p.title.lower())
        result['scope'].append({'area':key,'description':description,
             'structured_offers':count,'reviewed_reference_pages':references,
             'complete':False,
             'note':'Comprehensiveness cannot be certified without complete crawl and manual page-by-page review.'})
    return result
