"""Stage 7 live acceptance: live official website queries, no hidden cached prices."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_live import answer_live,OfficialFetcher

CHECKS = (
    ('internet', '100 takar moddhe 7 diner internet', 'live_public_listing'),
    ('combo', 'combo under 150 taka 30 days', 'live_public_listing'),
    ('minute', 'minute pack under 100 taka', 'live_public_listing'),
    ('sms', 'sms pack under 100 taka', 'live_public_listing'),
    ('special_sim', 'closed sim eligibility conditions', 'live_source_information'),
    ('customer_care', 'customer care information', 'live_source_information'),
)

def main():
    fetcher=OfficialFetcher()
    fails=[]
    for tag,question,want in CHECKS:
        result=answer_live(question,fetcher=fetcher,audit=False)
        assert result.mode in ('live_public_listing','live_source_information','unavailable','no_match','needs_clarification')
        source=result.citations[0] if result.citations else 'no citation'
        ok = result.mode == want and source.startswith('https://')
        if tag=='special_sim':
            ok = ok and any(x in result.message.lower() for x in
                ('selected customer','eligible','eligibility','নির্বাচিত','উপযুক্ত'))
            ok = ok and 'corporate information' not in result.message.lower()
        if tag=='customer_care':
            ok = ok and any(x in result.message.lower() for x in ('121','help line','helpline','হেল্পলাইন'))
        print(('PASS' if ok else 'FAIL')+f': {tag} | mode={result.mode} | source={source}')
        if not ok:
            print('  Reason/evidence:',result.message[:340].replace('\n',' | '))
            fails.append(tag)
    print('Result:', 'PASS: 6 categories verified live' if not fails else 'INCOMPLETE: '+', '.join(fails))
    print('Reminder: public listings are not subscriber-specific activation or eligibility checks.')
    return 1 if fails else 0

if __name__=='__main__':sys.exit(main())
