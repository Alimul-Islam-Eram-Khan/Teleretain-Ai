"""Unified Stage 5/6 prediction copilot + *fresh official website* knowledge.

Churn: existing validated local model and optional Gemini/Groq explanation.
Offers/eligibility: fresh official-only retrieval, deterministic output (no LLM price invention).
"""
from __future__ import annotations
from src.teletalk_live import answer_live

CHURN_WORDS=('churn', 'prediction', 'predict', 'risk score', 'চার্ন', 'প্রেডিকশন')


def ask_project(question: str, *, track=None, features=None, live=False, agent_tools=False, fetcher=None):
    if not isinstance(question, str): raise ValueError('Question must be text')
    if any(term in question.casefold() for term in CHURN_WORDS):
        if track is None or features is None:
            from src.teletalk_answer import KnowledgeAnswer
            return KnowledgeAnswer('needs_input','Iranian Track / IBM Benchmark',
                'Select Iranian or IBM public dataset and supply all validated inputs for a churn prediction.')
        from src.chatbot import ask_copilot
        return ask_copilot(question,track=track,features=features,live=live,agent_tools=agent_tools)
    return answer_live(question, fetcher=fetcher)
