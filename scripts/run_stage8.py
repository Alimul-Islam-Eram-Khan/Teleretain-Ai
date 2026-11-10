"""CLI demonstration: live public Teletalk recommendations. No LLM quota."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retention_recommender import RecommendationRequest, from_question, present, recommend


def main():
    ap = argparse.ArgumentParser(description='TeleRetain Stage 8 verified-offer matching')
    ap.add_argument('--question', type=str)
    ap.add_argument('--category', choices=['internet','combo','minute','sms','unlimited_weekend'])
    ap.add_argument('--budget', type=int)
    ap.add_argument('--validity-days', type=int, help='Exact required validity, in days')
    ap.add_argument('--at-least-days', type=int)
    ap.add_argument('--data-mb', type=int)
    ap.add_argument('--minutes', type=int)
    ap.add_argument('--sms', type=int)
    ap.add_argument('--sim-type', default='public', choices=['public','gen_z','professional','closed_sim','oporajita','mayer_hasi','agami','bornomala'])
    ap.add_argument('--sort', dest='sort_by', default='lowest_price', choices=['lowest_price','most_data','longest_validity','data_per_taka'])
    ap.add_argument('--language', default='banglish', choices=['english','bangla','banglish'])
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--json', action='store_true', help='Print source-audited structured output')
    args = ap.parse_args()
    try:
        if args.question:
            # Query mode and structured mode must not silently combine contradictory filters.
            structured = (args.category, args.budget, args.validity_days, args.at_least_days,
                          args.data_mb, args.minutes, args.sms)
            if any(x is not None for x in structured) or args.sim_type != 'public':
                ap.error('Use either --question or structured filters, not both.')
            result = from_question(args.question, language=args.language, limit=args.limit, sort_by=args.sort_by)
        else:
            request = RecommendationRequest(category=args.category, budget_bdt=args.budget,
                exact_validity_days=args.validity_days, min_validity_days=args.at_least_days,
                min_data_mb=args.data_mb, min_minutes=args.minutes, min_sms=args.sms,
                sim_type=args.sim_type, sort_by=args.sort_by, language=args.language, limit=args.limit)
            result = recommend(request)
    except (ValueError, TypeError) as exc:
        print(f'INPUT ERROR: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.json else present(result, args.language))
    return 0 if result.status in ('recommended','eligibility_verification_required','no_match','needs_clarification') else 1

if __name__=='__main__': sys.exit(main())
