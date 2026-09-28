"""
Tests for follow-up rounds: the coverage assessment of a round becomes the
search queries of the next round.
"""

from src.pipeline.followup import MAX_ASPECTS_PER_QUESTION, build_followup_queries
from src.pipeline.models import ResearchPlan, ResearchQuestion


def _plan():
    return ResearchPlan(questions=[
        ResearchQuestion(id="F1", question="How much power does the DGX B300 draw?",
                         search_terms=["DGX B300 power"], search_langs=["en", "de"],
                         search_terms_by_lang={"en": ["DGX B300 power"], "de": ["DGX B300 Stromaufnahme"]}),
        ResearchQuestion(id="F2", question="What does it cost?", search_terms=["DGX B300 price"],
                         search_langs=["en"], search_terms_by_lang={"en": ["DGX B300 price"]}),
        ResearchQuestion(id="F3", question="Who sells it in Germany?", search_terms=["DGX B300 reseller"],
                         search_langs=["en"], search_terms_by_lang={"en": ["DGX B300 reseller"]}),
        ResearchQuestion(id="F4", question="Is it rack-mountable?", search_terms=["DGX B300 rack"],
                         search_langs=["en"], search_terms_by_lang={"en": ["DGX B300 rack"]}),
    ])


def test_partial_question_gets_one_query_per_missing_aspect_and_language():
    fq = build_followup_queries(_plan(), {
        "F1": {"coverage": "partial", "missing_aspects": ["idle power", "- peak power."]},
    })
    assert len(fq) == 1 and fq[0]["question_id"] == "F1"
    assert fq[0]["search_terms"] == {
        "en": ["DGX B300 power idle power", "DGX B300 power peak power"],
        "de": ["DGX B300 Stromaufnahme idle power", "DGX B300 Stromaufnahme peak power"],
    }
    assert fq[0]["search_langs"] == ["en", "de"]


def test_answered_and_filter_blocked_questions_get_nothing():
    fq = build_followup_queries(_plan(), {
        "F1": {"coverage": "answered", "missing_aspects": ["x"]},
        "F2": {"coverage": "filter_blocked", "missing_aspects": ["y"]},
    })
    assert fq == []


def test_unanswered_without_aspects_searches_the_question_itself():
    fq = build_followup_queries(_plan(), {"F3": {"coverage": "unanswered", "missing_aspects": []}})
    assert fq[0]["search_terms"] == {"en": ["Who sells it in Germany?"]}


def test_partial_without_aspects_gets_nothing():
    """Nothing concrete to search for — better no query than a repeated one."""
    assert build_followup_queries(_plan(), {"F4": {"coverage": "partial", "missing_aspects": []}}) == []


def test_aspects_are_capped_and_queries_shortened():
    long_aspect = "one two three four five six seven eight nine ten eleven twelve"
    fq = build_followup_queries(_plan(), {
        "F2": {"coverage": "partial", "missing_aspects": [long_aspect, "b", "c", "d", "e"]},
    })
    terms = fq[0]["search_terms"]["en"]
    assert len(terms) == MAX_ASPECTS_PER_QUESTION
    assert all(len(t.split()) <= 10 for t in terms)


def test_questions_without_assessment_are_skipped():
    assert build_followup_queries(_plan(), {}) == []
