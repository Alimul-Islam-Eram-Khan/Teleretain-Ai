"""Recommended new UI interface: always fetch official website for price answers.

The legacy `answer_information` remains available solely for explicitly dated reviewed snapshots.
"""
from src.teletalk_live import answer_live as answer_public_offers
from src.teletalk_answer import KnowledgeAnswer as OfferAnswer, parse_question as parse_filters
from src.teletalk_answer import answer_information as answer_reviewed_information

def is_offer_question(question):
    if not isinstance(question,str) or not question.strip():return False
    from src.teletalk_answer import CATEGORY_WORDS
    low=question.lower()
    return any(term in low for words in CATEGORY_WORDS.values() for term in words) or any(
        term in low for term in ('offer','package','অফার','প্যাকেজ','eligibility','শর্ত'))
