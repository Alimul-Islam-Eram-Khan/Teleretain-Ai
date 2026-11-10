"""Deterministic offer matching and conservative eligibility answers.

Do not send raw web content, prices, or subscriber identifiers to an LLM.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import re
import unicodedata
from src.teletalk_kb import Offer, load_offers, load_info, load_conflicts

SIM_ALIASES={
    'gen_z': ('gen-z','gen z','genz','জেন জেড'),
    'professional': ('professional','profession','পেশাজীবী','teacher package','doctor package'),
    'closed_sim': ('closed sim','bondho sim','বন্ধ সিম','inactive sim'),
    'oporajita': ('oporajita','অপরাজিতা','women offer','female package','নারীদের প্যাকেজ'),
    'mayer_hasi': ('mayer hasi','mayer hashi','মায়ের হাসি','মায়ের হাসি'),
    'agami': ('agami','আগামী'),
    'bornomala': ('bornomala','বর্ণমালা'),
}
CATEGORY_WORDS={
    'combo': ('combo','combined','data minute','internet minute','কম্বো','ডাটা মিনিট'),
    'minute': ('minute','minutes','voice bundle','মিনিট'),
    'sms': ('sms','text bundle','এসএমএস'),
    'unlimited_weekend': ('unlimited','weekend','uninterrupted','আনলিমিটেড','উইকেন্ড'),
    'internet': ('internet','data','net pack','gb','mb','ইন্টারনেট','ডেটা','জিবি','এমবি'),
}
SPECIAL_WORDS=('eligible','eligibility','যোগ্য','conditions','condition','terms','শর্ত','registration','রেজিস্ট্রেশন','requirements','document','how to get','কিভাবে পাব','kibhabe pabo','পেতে')

@dataclass(frozen=True)
class KnowledgeAnswer:
    mode: str
    source_label: str
    message: str
    citations: tuple[str,...]=()
    offer_ids: tuple[str,...]=()
    provider: str='local-verified'


def digits(text: str) -> str:
    out=[]
    for c in text:
        try: out.append(str(unicodedata.digit(c)))
        except (ValueError,TypeError):out.append(c)
    return ''.join(out)


def parse_question(question: str)->dict:
    if not isinstance(question,str) or not question.strip() or len(question)>600:
        raise ValueError('Question must contain 1–600 characters.')
    q=digits(question).casefold().replace(',','')
    # Price requires a currency cue; do not confuse 7 days with budget 7.
    patterns=[
      r'(\d{1,5})\s*(?:টাকার|টাকা|takar|taka|tk)\s*(?:(?:er|এর)\s*)?(?:moddhe|মধ্যে|under|within|budget|কমে)',
      r'(?:under|within|below|budget|maximum|max|upto|up to|কমে|মধ্যে|বাজেট|moddhe|er moddhe)\s*(?:৳|tk\.?|bdt|taka|টাকা)?\s*(\d{1,5})(?:\s*(?:৳|tk\.?|bdt|taka|টাকা))?',
      r'(\d{1,5})\s*(?:৳|tk\.?|bdt|taka|টাকা)\s*(?:(?:er|এর)\s*)?(?:moddhe|মধ্যে|under|within|budget|কমে)',
      r'(?:৳|tk\.?|bdt|taka|টাকা)\s*(\d{1,5})(?=\b|\s|$)',
    ]
    price=None
    for pat in patterns:
        m=re.search(pat,q)
        if m:
            price=int(m.group(1)); break
    dm=re.search(r'(\d{1,3})\s*(?:days?|diner|din|দিন(?:ের)?)',q)
    day=int(dm.group(1)) if dm else None
    is_minimum=bool(re.search(r'(?:at least|minimum|min\.?|কমপক্ষে|অন্তত|atleast)',q))
    data=re.search(r'(\d+(?:\.\d+)?)\s*(gb|mb|জিবি|এমবি)',q)
    min_data=round(float(data.group(1))*(1024 if data.group(2) in ('gb','জিবি') else 1)) if data else None
    ussd=re.search(r'\*[0-9*]+#',q)
    minute=re.search(r'(\d{1,5})\s*(?:minutes?|min|মিনিট)',q)
    sms=re.search(r'(\d{1,5})\s*(?:sms|এসএমএস)',q)
    sim=next((k for k,aliases in SIM_ALIASES.items() if any(a in q for a in aliases)),None)
    category=next((k for k,words in CATEGORY_WORDS.items() if any(w in q for w in words)),None)
    # A combo query may name GB and minutes, so preference is combo; generic 'minutes' alone means minute.
    if (minute and data) or ('data' in q and 'minute' in q) or ('ইন্টারনেট' in q and 'মিনিট' in q): category='combo'
    return dict(budget_bdt=price,min_validity_days=day,exact_validity_days=day if day is not None and not is_minimum else None,ussd_code=ussd.group(0) if ussd else None,min_data_mb=min_data,
                min_minutes=int(minute.group(1)) if minute else None,
                min_sms=int(sms.group(1)) if sms else None,sim_type=sim,category=category,
                info_request=any(w in q for w in SPECIAL_WORDS))


def _matches(o:Offer, p:dict)->bool:
    if p.get('ussd_code') and o.ussd!=p['ussd_code']:return False
    if p['budget_bdt'] is not None and o.price_bdt>p['budget_bdt']:return False
    if p['min_validity_days'] is not None and (o.validity_days is None or o.validity_days<p['min_validity_days']):return False
    if p.get('exact_validity_days') is not None and o.validity_days!=p['exact_validity_days']:return False
    if p['min_data_mb'] is not None and (o.data_mb is None or o.data_mb<p['min_data_mb']):return False
    if p['min_minutes'] is not None and (o.minutes is None or o.minutes<p['min_minutes']):return False
    if p['min_sms'] is not None and (o.sms is None or o.sms<p['min_sms']):return False
    if p['category'] and o.category!=p['category']: return False
    if p['sim_type'] and o.sim_type not in ('public',p['sim_type']):return False
    if not p['sim_type'] and o.sim_type!='public':return False
    return True


def _format(o:Offer)->str:
    facts=[f'BDT {o.price_bdt}',f'{o.validity_days} days' if o.validity_days else 'validity not stated in days']
    if o.data_mb is not None: facts.append(f'{o.data_mb} MB')
    if o.minutes is not None:facts.append(f'{o.minutes} minutes')
    if o.sms is not None:facts.append(f'{o.sms} SMS')
    if o.ussd:facts.append('USSD '+o.ussd)
    facts.append('SIM '+o.sim_type)
    if o.eligibility_status=='requires_external_verification':facts.append('ELIGIBILITY NOT VERIFIED — confirm with Teletalk')
    if o.eligibility_conditions:facts.append('conditions: '+'; '.join(o.eligibility_conditions))
    if o.restrictions:facts.append('restrictions: '+'; '.join(o.restrictions))
    facts.append('activation: '+o.activation)
    return f'• {o.title}: '+ ' | '.join(facts)+f'\n  Official: {o.source_url} (snapshot {o.checked_on})'


def answer_offers(question:str,*,today:date|None=None,limit:int=6)->KnowledgeAnswer:
    label='Official Teletalk Public Knowledge (public website snapshot; not live subscriber eligibility)'
    try:
        if not (1<=limit<=20):raise ValueError('limit range')
        params=parse_question(question)
        if params['info_request'] and params['sim_type']:
            return answer_information(question,today=today)
        cat=load_offers(today=today)
        if params.get('ussd_code') and any(x.get('code')==params['ussd_code'] for x in load_conflicts()):
            return KnowledgeAnswer('unavailable',label,'This code/offer has conflicting official information and is quarantined. Confirm current details with Teletalk.')
        chosen=[o for o in cat.offers if _matches(o,params)]
        chosen.sort(key=lambda o:(o.price_bdt,-(o.data_mb or 0),-(o.minutes or 0),o.id))
        if not chosen:
            special=' Also tell me the SIM/package type if you want an eligibility-restricted offer.' if not params['sim_type'] else ''
            return KnowledgeAnswer('no_match',label,'No verified offer meets all specified conditions in this dated snapshot.'+special)
        chosen=chosen[:limit]
        lead=('Official public listing; not a live stock/activation or subscriber eligibility check. '
              'Always verify before purchasing. Special SIM eligibility must be confirmed by Teletalk.')
        return KnowledgeAnswer('grounded',label,lead+'\n'+'\n'.join(_format(o) for o in chosen),
                               tuple(dict.fromkeys(o.source_url for o in chosen)),tuple(o.id for o in chosen))
    except (ValueError,FileNotFoundError,KeyError,TypeError) as e:
        return KnowledgeAnswer('unavailable',label,f'Cannot safely assert current prices or availability: {type(e).__name__}. Re-review official sources; no offer claimed.')


def answer_information(question:str, *, today:date|None=None, retriever=None, limit:int=3)->KnowledgeAnswer:
    label='Official Teletalk Public Information (dated source excerpts)'
    try:
        p=parse_question(question)
        info=load_info(today=today)
        wanted=p['sim_type']
        # Direct exact page identification over dense model suggestions prevents wrong SIM descriptions.
        ids={'gen_z':'sim-gen-z','professional':'sim-professional','closed_sim':'sim-closed',
             'oporajita':'sim-oporajita','mayer_hasi':'sim-mayer-hasi','agami':'sim-agami','bornomala':'sim-bornomala'}
        matched=[row for row in info if row.id==ids.get(wanted)] if wanted else []
        if not matched:
            q=question.casefold()
            direct={'emergency':'service-emergency','advance balance':'service-emergency','জরুরি':'service-emergency',
                    'gravity':'postpaid-gravity','postpaid':'postpaid-gravity','rate cutter':'service-rate-cutter',
                    'shotoborsho':'sim-shotoborsho','শতবর্ষ':'sim-shotoborsho',
                    'esim':'service-esim','e-sim':'service-esim',
                    'customer care':'service-customer-care','roaming':'service-roaming','faq':'service-faq'}
            ident=next((v for k,v in direct.items() if k in q),None)
            if ident: matched=[row for row in info if row.id==ident]
        if not matched:
            if retriever is None:
                from src.teletalk_vector import retrieve
                retriever=retrieve
            results=retriever(question,top_k=limit)
            matched=[r['item'] for r in results if hasattr(r['item'],'body')]
            # Semantic retrieval cannot justify answering an unrelated question.
            stop={'what','how','when','can','you','the','a','and','is','are','please','tell','me',
                  'of','for','about','teletalk','offer','package','service','info','information','some',
                  'my','i','amar','amake','ki','kibhabe','sim','data','internet','প্যাকেজ','অফার',
                  'কিভাবে','কি','আমাকে','টেলিটক','এবং'}
            terms={t for t in re.findall(r'[a-z\u0980-\u09ff]{3,}',question.casefold()) if t not in stop}
            if not terms:
                matched=[]
            else:
                matched=[row for row in matched if any(t in (row.title+' '+row.body).casefold() for t in terms)]
        if not matched:
            return KnowledgeAnswer('no_match',label,'No reviewed official source answers this question. Check the official Teletalk website or support; no conditions invented.')
        matched=matched[:limit]
        lines=['Source-grounded excerpt; verify details with Teletalk, especially eligibility, pricing, dates and availability.']
        for x in matched:
            lines.append(f'• {x.title}: {x.body}\n  Official: {x.source_url} (reviewed {x.checked_on})')
        return KnowledgeAnswer('grounded',label,'\n'.join(lines),tuple(dict.fromkeys(x.source_url for x in matched)))
    except (OSError,ValueError,KeyError,TypeError,ImportError) as e:
        return KnowledgeAnswer('unavailable',label,f'Reviewed official knowledge unavailable ({type(e).__name__}). No unsupported claim served.')


def ask_teletalk(question:str,*,today:date|None=None)->KnowledgeAnswer:
    """Eligibility questions are source excerpts; offer recommendation is structured/deterministic."""
    try:
        p=parse_question(question)
    except ValueError:
        return KnowledgeAnswer('unsupported','Official Teletalk','Please ask a short public Teletalk question.')
    q=question.casefold()
    if p['info_request'] or (p['sim_type'] and not p['category'] and p['budget_bdt'] is None):
        return answer_information(question,today=today)
    if p['category'] or p['budget_bdt'] is not None or p.get('ussd_code') or any(w in q for w in ('offer','package','pack','অফার','প্যাকেজ','recharge')):
        return answer_offers(question,today=today)
    return answer_information(question,today=today)
