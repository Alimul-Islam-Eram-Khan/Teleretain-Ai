"""Stage 8 unified RESEARCH DEMO chat router; no changes to Stage 5/6/7 files.

Offer fitting: Stage 8 verified live official only.
Churn: Stage 5 public Iranian/IBM prediction + Stage 6 via Stage 7 bridge.
Service information: Stage 7 official live source reader.
"""
from __future__ import annotations
from src.retention_recommender import from_question, present, RecommendationResult
from src.teletalk_answer import parse_question
from src.teletalk_full_copilot import ask_project, CHURN_WORDS
from src.teletalk_live import answer_live

PACKAGE_CUES = ('offer','package','pack','bundle','প্যাকেজ','প্যাক','অফার','কম্বো',
                'internet','data','minute','sms','ইন্টারনেট','ডেটা','মিনিট','এসএমএস')


def ask_research_copilot(question: str, *, track=None, features=None, live_llm=False,
                         fetcher=None, language='banglish'):
    """Returns (mode, display_text, citations); never blends benchmark churn with offer prices."""
    if not isinstance(question,str) or not question.strip() or len(question)>600:
        return 'needs_clarification','Please send a short question.',()
    q=question.casefold()
    is_churn=any(token in q for token in CHURN_WORDS)
    try:
        p=parse_question(question)
    except ValueError:
        return 'needs_clarification','Invalid or ambiguous question.',()
    is_offer=bool(p['category'] or any(w in q for w in PACKAGE_CUES))
    if is_churn and is_offer:
        return ('needs_clarification',
                'Churn scores are public Iranian/IBM benchmark estimates, not actual Teletalk subscriber risk. '
                'Ask about benchmark churn or public Teletalk package matching separately.',())
    if is_churn:
        reply=ask_project(question,track=track,features=features,live=live_llm,fetcher=fetcher)
        return getattr(reply,'mode','prediction'),reply.message,tuple(getattr(reply,'citations',()))
    if is_offer or p['sim_type']:
        res=from_question(question,fetcher=fetcher,language=language)
        return res.status,present(res,language),res.citations
    reply=answer_live(question,fetcher=fetcher)
    return reply.mode,reply.message,tuple(reply.citations)
