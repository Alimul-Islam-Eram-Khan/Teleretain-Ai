"""Explicit human review lifecycle. NEVER auto-stamp old offer facts as refreshed."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from hashlib import sha256
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.teletalk_kb import STORE, OFFERS, INFO, REVIEWED, load_offers, load_info,catalog_fingerprint
CHECKLIST=STORE/'manual_source_revalidation.json'
LOG=STORE/'source_review_log.jsonl'

def prepare():
    offers=load_offers(allow_stale=True).offers
    info=load_info(allow_stale=True)
    items=[]
    for x in offers:
        items.append(dict(kind='offer',id=x.id,source_url=x.source_url,price_bdt=x.price_bdt,
                          title=x.title,reviewed=False,review_note=''))
    for x in info:
        items.append(dict(kind='info',id=x.id,source_url=x.source_url,title=x.title,
                          reviewed=False,review_note=''))
    content=dict(schema_version=1,original_fingerprint=catalog_fingerprint(),items=items,
                 instructions='Open EVERY source_url; compare its offer price, restrictions, eligibility and activation. Mark reviewed:true with specific review_note after checking. Unavailable/changed offers MUST be removed or corrected from approved catalog, then regenerate checklist; do not auto approve stale entries.')
    CHECKLIST.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return len(items)

def apply(*,explicit_ack:bool=False,reviewer:str=''):
    if not explicit_ack or len(reviewer.strip())<2: raise ValueError('Explicit reviewed-all acknowledgement and reviewer label required.')
    check=json.loads(CHECKLIST.read_text(encoding='utf-8'))
    if check['original_fingerprint']!=catalog_fingerprint():
        raise ValueError('Knowledge files changed since checklist preparation; regenerate and review.')
    offers=load_offers(allow_stale=True).offers
    pages=load_info(allow_stale=True)
    expected={('offer',x.id) for x in offers}|{('info',x.id) for x in pages}
    found={(item['kind'],item['id']) for item in check['items']}
    if expected != found or len(check['items'])!=len(expected):raise ValueError('Review items incomplete or duplicated.')
    if any(item.get('reviewed') is not True or len(item.get('review_note','').strip())<12 for item in check['items']):
        raise ValueError('All entries need explicit confirmation and a descriptive review note. Missing/unavailable entries must not be confirmed.')
    srcmap={('offer',x.id):x.source_url for x in offers}|{('info',x.id):x.source_url for x in pages}
    for item in check['items']:
        key=(item['kind'],item['id'])
        if item['source_url']!=srcmap[key]:raise ValueError('Official source changed since checklist; re-review.')
        if item['kind']=='offer':
            by_id={x.id:x.price_bdt for x in offers}
            if item['price_bdt']!=by_id[item['id']]:raise ValueError('Price changed since original checklist; re-review.')
    today=datetime.now(ZoneInfo('Asia/Dhaka')).date().isoformat()
    backup=STORE/'review_backups'/datetime.now(ZoneInfo('Asia/Dhaka')).strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(OFFERS,backup/OFFERS.name);shutil.copy2(INFO,backup/INFO.name)
    if REVIEWED.exists():shutil.copy2(REVIEWED,backup/REVIEWED.name)
    # Update only reviewer-attested dates. No data values changed.
    raw_offers=json.loads(OFFERS.read_text(encoding='utf-8'))
    raw_info=json.loads(INFO.read_text(encoding='utf-8'))
    raw_offers['snapshot_compiled_on']=today
    raw_info['snapshot_compiled_on']=today
    for x in raw_offers['offers']:x['checked_on']=today
    for x in raw_info['pages']:x['checked_on']=today
    documents=[(OFFERS,raw_offers),(INFO,raw_info)]
    if REVIEWED.exists():
        raw_reviewed=json.loads(REVIEWED.read_text(encoding='utf-8'))
        for x in raw_reviewed['pages']:x['checked_on']=today
        documents.append((REVIEWED,raw_reviewed))
    for path,payload in documents:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
    log={'reviewed_at':today,'reviewer_label':reviewer,'count':len(expected),'backup':str(backup),
         'review_items_sha256':sha256(CHECKLIST.read_bytes()).hexdigest(),
         'notice':'Reviewer manually attested; program cannot certify website content.'}
    with LOG.open('a',encoding='utf-8') as out:out.write(json.dumps(log,ensure_ascii=False)+'\n')
    return log

if __name__=='__main__':
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true')
    g.add_argument('--apply',action='store_true')
    p.add_argument('--i-checked-every-record',action='store_true')
    p.add_argument('--reviewer',default='')
    a=p.parse_args()
    if a.prepare:
        print('Created manual review checklist for',prepare(),'entries:',CHECKLIST)
        print('Never set reviewed=true without checking each record at its exact official source.')
    else:
        print('Applied human attested revalidation:',apply(explicit_ack=a.i_checked_every_record,reviewer=a.reviewer))
        print('Rebuild FAISS after catalog update.')
