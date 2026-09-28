"""
Tests for the literature search service and the query methods of the
LiteratureAPIClient.

Tests:
1. SearchResult data type (dedupe_key, merge_from, relevance_score)
2. dedupe_results helper (DOI + title similarity)
3. parser methods of the APIs (OpenAlex, S2, arXiv)
4. _reconstruct_openalex_abstract (inverted index → string)
5. LiteratureSearchService with MockAPIClient
   - multi_source_query
   - expand_from_seeds (serial)
   - search_with_expansion (combined + PRISMA log)

httpx is mocked, so the tests are pure unit tests without network.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# mock httpx and rate_limiter before literature_apis is imported
if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()
if "src.connectors.rate_limiter" not in sys.modules:
    _rl_mock = MagicMock()
    _rl_mock.get_rate_limiter = MagicMock(return_value=MagicMock())
    sys.modules["src.connectors.rate_limiter"] = _rl_mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectors.literature_search import (  # noqa: E402
    LiteratureSearchService,
    SearchResult,
    dedupe_results,
    _normalize_title,
    _titles_match,
)


# ─── Tests for SearchResult ────────────────────────────────────────

def test_search_result_dedupe_key_doi():
    """A DOI-based de-duplication key prefers the DOI."""
    print("Test: SearchResult dedupe_key with DOI ...", end=" ")
    r = SearchResult(
        doi="10.1234/abc",
        title="Some Paper",
    )
    # there is no dedupe_key, but merge_from compares the DOI
    # so we test the merge_from behaviour here
    r2 = SearchResult(
        doi="10.1234/abc",
        title="Some Paper — Alternate Title",
        abstract="Longer abstract text",
        cited_by_count=42,
    )
    r.merge_from(r2)
    assert r.cited_by_count == 42
    assert len(r.abstract) > 0
    print("✓")


def test_search_result_merge_from_longer_abstract():
    """merge_from prefers longer abstracts."""
    print("Test: merge_from Abstract-Merge ...", end=" ")
    a = SearchResult(title="X", abstract="short")
    b = SearchResult(title="X", abstract="this is a much longer abstract with more content")
    a.merge_from(b)
    assert a.abstract == "this is a much longer abstract with more content"
    print("✓")


def test_search_result_merge_from_discovered_via():
    """merge_from merges discovered_via as a list without duplicates."""
    print("Test: merge_from discovered_via ...", end=" ")
    a = SearchResult(title="X", discovered_via=["query:foo"])
    b = SearchResult(title="X", discovered_via=["query:bar"])
    a.merge_from(b)
    assert "query:foo" in a.discovered_via
    assert "query:bar" in a.discovered_via
    assert len(a.discovered_via) == 2

    # add a duplicate
    c = SearchResult(title="X", discovered_via=["query:foo"])
    a.merge_from(c)
    assert len(a.discovered_via) == 2  # no additional entry
    print("✓")


def test_search_result_merge_from_citations_max():
    """merge_from takes the maximum of the citation counts."""
    print("Test: merge_from cited_by_count max ...", end=" ")
    a = SearchResult(title="X", cited_by_count=10)
    b = SearchResult(title="X", cited_by_count=25)
    a.merge_from(b)
    assert a.cited_by_count == 25
    print("✓")


def test_search_result_merge_from_peer_reviewed():
    """merge_from: True overrides None, but False does not override True."""
    print("Test: merge_from peer_reviewed ...", end=" ")
    a = SearchResult(title="X", is_peer_reviewed=None)
    b = SearchResult(title="X", is_peer_reviewed=True)
    a.merge_from(b)
    assert a.is_peer_reviewed is True

    # True stays True, even if another says False
    c = SearchResult(title="X", is_peer_reviewed=True)
    d = SearchResult(title="X", is_peer_reviewed=False)
    c.merge_from(d)
    assert c.is_peer_reviewed is True
    print("✓")


def test_search_result_relevance_score_no_boost():
    """relevance_score is the plain citation count, NO discovery boost."""
    print("Test: relevance_score without a discovery boost ...", end=" ")
    a = SearchResult(
        title="X",
        cited_by_count=10,
        discovered_via=["query:foo"],
    )
    b = SearchResult(
        title="X",
        cited_by_count=10,
        discovered_via=["query:foo", "forward_from:10.x", "backward_from:10.y"],
    )
    # both have the same score, although b was discovered several times
    assert a.relevance_score == b.relevance_score == 10.0
    print("✓")


# ─── Tests for de-duplication ──────────────────────────────────────

def test_dedupe_by_doi():
    """Exact DOI matches are merged."""
    print("Test: dedupe per DOI ...", end=" ")
    results = [
        SearchResult(doi="10.1234/abc", title="A", cited_by_count=10,
                     discovered_via=["q1"]),
        SearchResult(doi="10.1234/abc", title="A (alternate)",
                     cited_by_count=15, discovered_via=["q2"]),
    ]
    deduped = dedupe_results(results)
    assert len(deduped) == 1
    assert deduped[0].cited_by_count == 15  # Maximum
    assert "q1" in deduped[0].discovered_via
    assert "q2" in deduped[0].discovered_via
    print("✓")


def test_dedupe_by_title_similarity():
    """Similar titles without a DOI are merged too."""
    print("Test: dedupe by title similarity ...", end=" ")
    results = [
        SearchResult(title="Large Language Models for Science", discovered_via=["q1"]),
        SearchResult(title="Large language models for science.", discovered_via=["q2"]),
    ]
    deduped = dedupe_results(results)
    assert len(deduped) == 1, f"Expected 1, got {len(deduped)}"
    print("✓")


def test_dedupe_different_titles_kept():
    """Different titles are kept."""
    print("Test: dedupe different titles ...", end=" ")
    results = [
        SearchResult(title="Attention is All You Need"),
        SearchResult(title="BERT Pre-training"),
        SearchResult(title="GPT-3 Language Models"),
    ]
    deduped = dedupe_results(results)
    assert len(deduped) == 3
    print("✓")


def test_dedupe_doi_and_title_mixed():
    """Entries with a DOI and title-based entries are handled correctly."""
    print("Test: dedupe DOI + title mixed ...", end=" ")
    results = [
        SearchResult(doi="10.1/a", title="Paper A"),
        SearchResult(doi="10.2/b", title="Paper B"),
        SearchResult(title="Paper A"),  # duplicate of #1, but without a DOI
    ]
    deduped = dedupe_results(results)
    # the third is matched against paper A (from the first) and merged
    assert len(deduped) == 2, f"Expected 2, got {len(deduped)}"
    print("✓")


# ─── Normalisation tests ─────────────────────────────────────────

def test_normalize_title_basic():
    """Normalisation: lower case, punctuation removed, whitespace collapsed."""
    print("Test: _normalize_title ...", end=" ")
    assert _normalize_title("Large Language Models!") == "large language models"
    assert _normalize_title("  Multi   Spaces  ") == "multi spaces"
    assert _normalize_title("Python 3.11 Features") == "python 3 11 features"
    print("✓")


def test_titles_match_identical():
    print("Test: _titles_match identical ...", end=" ")
    assert _titles_match("Foo Bar", "Foo Bar")
    print("✓")


def test_titles_match_similar():
    print("Test: _titles_match similar ...", end=" ")
    # same title, different punctuation
    assert _titles_match(
        "Large Language Models for Science",
        "Large Language Models for Science.",
    )
    print("✓")


def test_titles_match_different():
    print("Test: _titles_match different ...", end=" ")
    assert not _titles_match("Paper A", "Paper B")
    assert not _titles_match("", "Something")
    print("✓")


# ─── Mock API client for service tests ─────────────────────────────

class MockAPIClient:
    """Simulates LiteratureAPIClient without HTTP calls.

    Every method returns predefined SearchResults. Calls are recorded in
    self.calls for assertions.
    """

    def __init__(self):
        self.calls: list[tuple] = []
        self.query_results: dict[str, list[SearchResult]] = {
            "openalex": [],
            "semantic_scholar": [],
            "arxiv": [],
        }
        self.citers_results: dict[str, list[SearchResult]] = {}
        self.refs_results: dict[str, list[SearchResult]] = {}

    async def query_openalex(self, query: str, limit: int = 25,
                              year_from=None, year_to=None,
                              peer_reviewed_only: bool = False):
        self.calls.append(("query_openalex", query, limit, year_from,
                          year_to, peer_reviewed_only))
        return list(self.query_results.get("openalex", []))

    async def query_semantic_scholar(self, query: str, limit: int = 25,
                                      year_from=None, year_to=None):
        self.calls.append(("query_semantic_scholar", query, limit,
                          year_from, year_to))
        return list(self.query_results.get("semantic_scholar", []))

    async def query_arxiv(self, query: str, limit: int = 25,
                           year_from=None, year_to=None):
        self.calls.append(("query_arxiv", query, limit, year_from, year_to))
        return list(self.query_results.get("arxiv", []))

    async def get_citers_openalex(self, doi: str, limit: int = 25):
        self.calls.append(("get_citers", doi, limit))
        return list(self.citers_results.get(doi, []))

    async def get_references_openalex(self, doi: str, limit: int = 25):
        self.calls.append(("get_references", doi, limit))
        return list(self.refs_results.get(doi, []))


# ─── Service-Tests ─────────────────────────────────────────────────

async def test_service_multi_source_parallel():
    """multi_source_query calls all three APIs in parallel."""
    print("Test: multi_source_query parallel ...", end=" ")
    mock = MockAPIClient()
    mock.query_results["openalex"] = [
        SearchResult(doi="10.1/oa", title="OpenAlex Paper",
                     cited_by_count=100, sources=["openalex"]),
    ]
    mock.query_results["semantic_scholar"] = [
        SearchResult(doi="10.2/s2", title="S2 Paper",
                     cited_by_count=50, sources=["semantic_scholar"]),
    ]
    mock.query_results["arxiv"] = [
        SearchResult(doi="10.48550/arXiv.2301.12345",
                     title="arXiv Paper", sources=["arxiv"]),
    ]

    service = LiteratureSearchService(mock)
    results = await service.multi_source_query("test query", limit=10)

    # all three sources called
    call_names = [c[0] for c in mock.calls]
    assert "query_openalex" in call_names
    assert "query_semantic_scholar" in call_names
    assert "query_arxiv" in call_names

    # 3 de-duplicated → 3
    assert len(results) == 3

    # sorted by cited_by_count, descending
    assert results[0].cited_by_count >= results[1].cited_by_count
    assert results[0].doi == "10.1/oa"  # Top
    print("✓")


async def test_service_dedupe_across_sources():
    """If the same DOI appears in several sources: de-duplicated."""
    print("Test: multi_source_query Dedupe ...", end=" ")
    mock = MockAPIClient()
    mock.query_results["openalex"] = [
        SearchResult(doi="10.1/same", title="Same Paper",
                     cited_by_count=100, sources=["openalex"],
                     abstract="short"),
    ]
    mock.query_results["semantic_scholar"] = [
        SearchResult(doi="10.1/same", title="Same Paper",
                     cited_by_count=105, sources=["semantic_scholar"],
                     abstract="much longer abstract from S2 here"),
    ]
    mock.query_results["arxiv"] = []

    service = LiteratureSearchService(mock)
    results = await service.multi_source_query("test", limit=10)

    assert len(results) == 1  # de-duplicated
    r = results[0]
    # maximum of the citations
    assert r.cited_by_count == 105
    # the longer abstract won
    assert "much longer" in r.abstract
    # both sources in the sources field
    assert "openalex" in r.sources
    assert "semantic_scholar" in r.sources
    print("✓")


async def test_service_min_citations_filter():
    """min_citations filters results below the threshold."""
    print("Test: min_citations-Filter ...", end=" ")
    mock = MockAPIClient()
    mock.query_results["openalex"] = [
        SearchResult(doi="10.1/high", title="High", cited_by_count=50),
        SearchResult(doi="10.2/low", title="Low", cited_by_count=2),
        SearchResult(doi="10.3/med", title="Med", cited_by_count=10),
    ]

    service = LiteratureSearchService(mock)
    results = await service.multi_source_query(
        "test", limit=10, min_citations=5, sources=["openalex"],
    )

    assert len(results) == 2
    assert all(r.cited_by_count >= 5 for r in results)
    print("✓")


async def test_service_peer_reviewed_filter():
    """peer_reviewed_only filters on True."""
    print("Test: peer_reviewed-Filter ...", end=" ")
    mock = MockAPIClient()
    mock.query_results["openalex"] = [
        SearchResult(doi="10.1/pr", title="PR", is_peer_reviewed=True,
                     cited_by_count=10),
        SearchResult(doi="10.2/npr", title="NPR", is_peer_reviewed=False,
                     cited_by_count=20),
        SearchResult(doi="10.3/unk", title="UNK", is_peer_reviewed=None,
                     cited_by_count=30),
    ]

    service = LiteratureSearchService(mock)
    results = await service.multi_source_query(
        "test", limit=10,
        peer_reviewed_only=True,
        sources=["openalex"],
    )

    assert len(results) == 1
    assert results[0].is_peer_reviewed is True
    print("✓")


async def test_service_expand_from_seeds():
    """expand_from_seeds runs serially and covers both directions."""
    print("Test: expand_from_seeds serially ...", end=" ")
    mock = MockAPIClient()
    mock.citers_results = {
        "10.1/seed": [
            SearchResult(doi="10.10/citer1", title="Citer 1",
                         cited_by_count=5),
            SearchResult(doi="10.11/citer2", title="Citer 2",
                         cited_by_count=3),
        ],
    }
    mock.refs_results = {
        "10.1/seed": [
            SearchResult(doi="10.20/ref1", title="Ref 1",
                         cited_by_count=50),
            SearchResult(doi="10.21/ref2", title="Ref 2",
                         cited_by_count=30),
        ],
    }

    service = LiteratureSearchService(mock)
    results = await service.expand_from_seeds(
        seed_dois=["10.1/seed"],
        max_per_seed=10,
    )

    # both directions called
    call_names = [c[0] for c in mock.calls]
    assert call_names == ["get_citers", "get_references"]  # serial!

    # 4 results, de-duplicated
    assert len(results) == 4

    # all have seed_distance=1
    assert all(r.seed_distance == 1 for r in results)

    # all have a matching discovered_via
    forward = [r for r in results if any("forward_from" in d for d in r.discovered_via)]
    backward = [r for r in results if any("backward_from" in d for d in r.discovered_via)]
    assert len(forward) == 2
    assert len(backward) == 2
    print("✓")


async def test_service_expand_forward_only():
    """expand_from_seeds with directions=['forward']."""
    print("Test: expand_from_seeds forward only ...", end=" ")
    mock = MockAPIClient()
    mock.citers_results = {
        "10.1/a": [SearchResult(doi="10.10/c1", title="Citer")],
    }

    service = LiteratureSearchService(mock)
    results = await service.expand_from_seeds(
        seed_dois=["10.1/a"],
        directions=["forward"],
    )

    call_names = [c[0] for c in mock.calls]
    assert call_names == ["get_citers"]
    assert "get_references" not in call_names
    assert len(results) == 1
    print("✓")


async def test_service_search_with_expansion_combined():
    """search_with_expansion: query + seeds combined."""
    print("Test: search_with_expansion combined ...", end=" ")
    mock = MockAPIClient()
    mock.query_results["openalex"] = [
        SearchResult(doi="10.1/q1", title="Query Hit", cited_by_count=20),
    ]
    mock.query_results["semantic_scholar"] = []
    mock.query_results["arxiv"] = []
    mock.citers_results = {
        "10.99/seed": [
            SearchResult(doi="10.2/c1", title="Citer", cited_by_count=5),
        ],
    }
    mock.refs_results = {
        "10.99/seed": [
            SearchResult(doi="10.3/r1", title="Ref", cited_by_count=50),
        ],
    }

    service = LiteratureSearchService(mock)
    results, log = await service.search_with_expansion(
        query="test query",
        seed_dois=["10.99/seed"],
    )

    assert len(results) == 3
    assert log["n_from_query"] == 1
    assert log["n_from_expansion"] == 2
    assert log["n_after_dedupe"] >= 3
    assert log["query_used"] == "test query"
    assert log["seeds_used"] == ["10.99/seed"]

    # sorted by citations
    assert results[0].cited_by_count == 50  # Ref1 is cited most often
    print("✓")


async def test_service_search_only_query():
    """search_with_expansion without seeds → query only."""
    print("Test: search_with_expansion query only ...", end=" ")
    mock = MockAPIClient()
    mock.query_results["openalex"] = [
        SearchResult(doi="10.1/q1", title="Q1", cited_by_count=10),
    ]

    service = LiteratureSearchService(mock)
    results, log = await service.search_with_expansion(query="test")

    assert len(results) == 1
    assert log["n_from_query"] == 1
    assert log["n_from_expansion"] == 0
    # get_citers and get_references NOT called
    call_names = [c[0] for c in mock.calls]
    assert "get_citers" not in call_names
    assert "get_references" not in call_names
    print("✓")


async def test_service_search_only_seeds():
    """search_with_expansion without a query → expansion only."""
    print("Test: search_with_expansion seeds only ...", end=" ")
    mock = MockAPIClient()
    mock.citers_results = {"10.1/s": [SearchResult(doi="10.2/x", title="X")]}
    mock.refs_results = {"10.1/s": [SearchResult(doi="10.3/y", title="Y")]}

    service = LiteratureSearchService(mock)
    results, log = await service.search_with_expansion(seed_dois=["10.1/s"])

    assert len(results) == 2
    assert log["n_from_query"] == 0
    assert log["n_from_expansion"] == 2
    # query_* NOT called
    call_names = [c[0] for c in mock.calls]
    assert "query_openalex" not in call_names
    print("✓")


# ─── Parser tests (for the static helpers in LiteratureAPIClient) ──

def test_reconstruct_openalex_abstract():
    """OpenAlex' inverted_index is converted into a string correctly."""
    print("Test: _reconstruct_openalex_abstract ...", end=" ")
    # lazy import, because LiteratureAPIClient needs httpx
    from src.connectors.literature_apis import LiteratureAPIClient
    # inverted index: {word: [positions]}
    inverted = {
        "This": [0],
        "is": [1, 4],
        "a": [2],
        "test": [3],
        "only": [5],
        "test.": [6],
    }
    # expected order: position 0=This, 1=is, 2=a, 3=test, 4=is, 5=only, 6=test.
    result = LiteratureAPIClient._reconstruct_openalex_abstract(inverted)
    assert result == "This is a test is only test."
    print("✓")


def test_reconstruct_empty_abstract():
    """Empty or None index → empty string."""
    print("Test: _reconstruct empty ...", end=" ")
    from src.connectors.literature_apis import LiteratureAPIClient
    assert LiteratureAPIClient._reconstruct_openalex_abstract(None) == ""
    assert LiteratureAPIClient._reconstruct_openalex_abstract({}) == ""
    assert LiteratureAPIClient._reconstruct_openalex_abstract("not a dict") == ""
    print("✓")


def test_openalex_item_to_search_result():
    """An OpenAlex item is mapped to a SearchResult correctly."""
    print("Test: _openalex_item_to_search_result ...", end=" ")
    from src.connectors.literature_apis import LiteratureAPIClient
    client = LiteratureAPIClient.__new__(LiteratureAPIClient)  # without __init__

    item = {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/xyz",
        "display_name": "A Great Paper",
        "publication_year": 2023,
        "cited_by_count": 42,
        "type": "article",
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ],
        "primary_location": {
            "source": {
                "display_name": "Nature",
                "type": "journal",
            }
        },
        "abstract_inverted_index": {
            "This": [0], "paper": [1], "explains": [2], "stuff.": [3],
        },
    }
    result = client._openalex_item_to_search_result(item)
    assert result is not None
    assert result.doi == "10.1234/xyz"
    assert result.title == "A Great Paper"
    assert result.year == 2023
    assert result.cited_by_count == 42
    assert result.is_peer_reviewed is True  # journal-Typ
    assert result.venue == "Nature"
    assert len(result.authors) == 2
    assert "Alice Smith" in result.authors
    assert result.abstract == "This paper explains stuff."
    assert "openalex" in result.sources
    print("✓")


def test_openalex_item_repository_not_peer_reviewed():
    """repository-source-type → is_peer_reviewed=False."""
    print("Test: OpenAlex repository-Typ ...", end=" ")
    from src.connectors.literature_apis import LiteratureAPIClient
    client = LiteratureAPIClient.__new__(LiteratureAPIClient)

    item = {
        "id": "https://openalex.org/W1",
        "display_name": "Preprint",
        "publication_year": 2023,
        "primary_location": {
            "source": {"display_name": "arXiv", "type": "repository"},
        },
    }
    result = client._openalex_item_to_search_result(item)
    assert result.is_peer_reviewed is False
    print("✓")


def test_s2_item_to_search_result():
    """An S2 item is mapped correctly."""
    print("Test: _s2_item_to_search_result ...", end=" ")
    from src.connectors.literature_apis import LiteratureAPIClient

    item = {
        "paperId": "abc123",
        "title": "Some Paper",
        "year": 2022,
        "venue": "ACL",
        "externalIds": {"DOI": "10.9/test"},
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "abstract": "Full abstract text here.",
        "citationCount": 15,
        "publicationTypes": ["JournalArticle"],
    }
    result = LiteratureAPIClient._s2_item_to_search_result(item)
    assert result.doi == "10.9/test"
    assert result.title == "Some Paper"
    assert result.year == 2022
    assert result.cited_by_count == 15
    assert result.is_peer_reviewed is True  # JournalArticle
    assert result.abstract == "Full abstract text here."
    assert "semantic_scholar" in result.sources
    print("✓")


def test_arxiv_full_parser():
    """arXiv XML is parsed correctly, including the abstract."""
    print("Test: _parse_arxiv_response_full ...", end=" ")
    from src.connectors.literature_apis import LiteratureAPIClient

    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>A Test Paper
    About LLMs</title>
    <summary>
    This paper presents a novel approach to language models.
    We show significant improvements.
    </summary>
    <published>2023-01-15T12:00:00Z</published>
    <author><name>First Author</name></author>
    <author><name>Second Author</name></author>
  </entry>
</feed>'''

    results = LiteratureAPIClient._parse_arxiv_response_full(xml)
    assert len(results) == 1
    r = results[0]
    assert "Test Paper" in r["title"]
    assert r["year"] == 2023
    assert "novel approach" in r["abstract"]
    assert r["arxiv_id"] == "2301.12345"
    assert len(r["authors"]) == 2
    assert r["doi"] == "10.48550/arXiv.2301.12345"
    print("✓")


# ─── Test-Runner ────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Literature Search Service Tests")
    print("=" * 60)

    sync_tests = [
        test_search_result_dedupe_key_doi,
        test_search_result_merge_from_longer_abstract,
        test_search_result_merge_from_discovered_via,
        test_search_result_merge_from_citations_max,
        test_search_result_merge_from_peer_reviewed,
        test_search_result_relevance_score_no_boost,
        test_dedupe_by_doi,
        test_dedupe_by_title_similarity,
        test_dedupe_different_titles_kept,
        test_dedupe_doi_and_title_mixed,
        test_normalize_title_basic,
        test_titles_match_identical,
        test_titles_match_similar,
        test_titles_match_different,
        test_reconstruct_openalex_abstract,
        test_reconstruct_empty_abstract,
        test_openalex_item_to_search_result,
        test_openalex_item_repository_not_peer_reviewed,
        test_s2_item_to_search_result,
        test_arxiv_full_parser,
    ]
    async_tests = [
        test_service_multi_source_parallel,
        test_service_dedupe_across_sources,
        test_service_min_citations_filter,
        test_service_peer_reviewed_filter,
        test_service_expand_from_seeds,
        test_service_expand_forward_only,
        test_service_search_with_expansion_combined,
        test_service_search_only_query,
        test_service_search_only_seeds,
    ]

    failed = 0
    for t in sync_tests:
        try:
            t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    for t in async_tests:
        try:
            await t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total = len(sync_tests) + len(async_tests)
    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{total} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
