"""Stage 7 RAG APIs: live official retrieval vs a separately labelled dated FAISS snapshot.

IMPORTANT: FAISS similarity is retrieval, not evidence that a package is currently active.
For customer-facing prices use retrieve_public_offers() / answer_live(), NOT the dated snapshot.
"""
from __future__ import annotations
from dataclasses import dataclass
from src.teletalk_live import answer_live
from src.teletalk_vector import build as build_reviewed_index, retrieve as retrieve_reviewed_knowledge
from src.teletalk_kb import Offer, InfoPage, document_text

@dataclass(frozen=True)
class ReviewedDocument:
    item: Offer | InfoPage
    similarity: float
    document: str
    @property
    def offer(self):
        return self.item if isinstance(self.item,Offer) else None


def retrieve_public_offers(question: str, *, fetcher=None, limit=6):
    """Returns source-labelled *live* KnowledgeAnswer; no LLM generates prices."""
    return answer_live(question,fetcher=fetcher,limit=limit)


def retrieve_reviewed_snapshot(question: str, *, top_k=5, **kwargs) -> list[ReviewedDocument]:
    """FAISS research context ONLY; do not use these historic amounts as live prices."""
    return [ReviewedDocument(x['item'],x['score'],document_text(x['item']))
            for x in retrieve_reviewed_knowledge(question,top_k=top_k,**kwargs)]
