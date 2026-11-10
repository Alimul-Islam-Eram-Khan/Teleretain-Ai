"""Category/eligibility-aware deterministic recommendation (no language model)."""
from collections.abc import Iterable
from src.teletalk_kb import Offer

def recommend_verified_offers(offers:Iterable[Offer],*,budget_bdt=None,min_validity_days=None,
                              min_data_mb=None,min_minutes=None,min_sms=None,exact_validity_days=None,sim_type=None,
                              category=None,limit=5)->list[Offer]:
    from src.teletalk_answer import _matches
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=20:
        raise ValueError('limit must be 1–20.')
    for name,val in (('budget',budget_bdt),('validity',min_validity_days),('data',min_data_mb),('minutes',min_minutes),('sms',min_sms)):
        if val is not None and (type(val) is not int or val<0): raise ValueError(f'Invalid {name} filter.')
    if sim_type is not None and not isinstance(sim_type,str):raise ValueError('Invalid SIM type.')
    if category is not None and not isinstance(category,str):raise ValueError('Invalid category.')
    p=dict(budget_bdt=budget_bdt,min_validity_days=min_validity_days,exact_validity_days=exact_validity_days,min_data_mb=min_data_mb,
           min_minutes=min_minutes,min_sms=min_sms,sim_type=sim_type,category=category)
    results=[o for o in offers if _matches(o,p)]
    return sorted(results,key=lambda o:(o.price_bdt,-(o.data_mb or 0),o.id))[:limit]


def recommend_live_public_offers(question: str, *, fetcher=None, limit=6):
    """Public-facing deterministic recommendation: re-fetch official source for every request.

    Does not claim individual eligibility or live operator-side activation.
    """
    from src.teletalk_live import answer_live
    return answer_live(question,fetcher=fetcher,limit=limit)
