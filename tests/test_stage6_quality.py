"""Offline quality regressions: never accept incomplete or ungrounded numeric responses."""

from src import chatbot as cb


def test_cut_off_banglish_is_rejected():
    assert not cb._is_safe_llm_text(
        "Public non-Teletalk dataset bebohar kore model ti apnar churn howar sh"
    )


def test_cut_off_english_is_rejected():
    assert not cb._is_safe_llm_text(
        "Based on a public benchmark dataset, the model estimates a low likelihood of"
    )


def test_completed_sentence_accepted():
    assert cb._is_safe_llm_text(
        "The model flags churn for this public dataset. It does not predict real Teletalk customers."
    )


def test_bangla_punctuation_accepted():
    assert cb._is_safe_llm_text("এটি পাবলিক ডেটাসেটের মডেলের অনুমান।")


def test_new_numbers_rejected_even_when_complete():
    assert not cb._is_safe_llm_text("The score is 98 percent.")


def test_unverified_packages_rejected():
    assert not cb._is_safe_llm_text("Buy an official package now.")
