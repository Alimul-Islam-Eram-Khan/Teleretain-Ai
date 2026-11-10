"""Optional Stage 8 demonstration bridge.

Churn prediction and public offer matching are strictly SEPARATE evidence tracks.
Combining their display does not validate Teletalk-specific risk, retention impact,
or subscriber eligibility. This module never adjusts the Stage 5 model.
"""
from __future__ import annotations
from dataclasses import dataclass
from src.retention_recommender import RecommendationRequest, RecommendationResult, recommend

@dataclass(frozen=True)
class ResearchDemo:
    public_dataset_prediction: dict
    separate_public_offer_matches: RecommendationResult
    warning: str = (
        'Two independent demonstrations: the churn score belongs to a non-Teletalk '
        'public dataset, while offers are Teletalk public website listings. '
        'They are NOT a personalized recommendation for a Teletalk customer.'
    )


def demo_independent_tracks(track: str, features: dict, preferences: RecommendationRequest,
                            *, fetcher=None, offer_loader=None) -> ResearchDemo:
    if track not in ('iranian', 'ibm') or not isinstance(features, dict):
        raise ValueError('Use a valid explicit non-Teletalk public dataset track and features.')
    if not isinstance(preferences, RecommendationRequest):
        raise TypeError('Validated offer preferences required')
    from src.routing import predict_customer  # Stage 5 validation runs BEFORE offer lookup.
    outcome = predict_customer({'track':track,'features':features})
    kwargs = {'fetcher':fetcher}
    if offer_loader is not None: kwargs['offer_loader'] = offer_loader
    matches = recommend(preferences,**kwargs)
    return ResearchDemo(outcome.model_dump(), matches)
