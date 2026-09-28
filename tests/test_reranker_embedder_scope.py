"""
Tests for the reranker, embedder and search scope.

Focus: the FAIL-OPEN behaviour. These building blocks are quality
improvements, not hard dependencies — missing configuration or a dead
endpoint must NEVER break the pipeline.
"""

import asyncio
import os
import unittest

import tests.conftest  # noqa: F401

from src.llm.reranker import Reranker, build_reranker_from_pipeline_config
from src.llm.embedder import Embedder, cosine
from src.pipeline.classifiers.search_scope import classify_search_scope
from src.config import PipelineConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(coro):
    return asyncio.run(coro)


class TestRerankerFailOpen(unittest.TestCase):
    def test_no_api_key_is_unavailable(self):
        rr = Reranker(base_url="https://x/rerank", model="m", api_key="")
        self.assertFalse(rr.available)

    def test_unavailable_rank_returns_identity(self):
        rr = Reranker("", "m", "")
        out = run(rr.rank("frage", ["a", "b", "c"]))
        self.assertEqual([i for i, _ in out], [0, 1, 2])

    def test_unavailable_order_keeps_items(self):
        rr = Reranker("", "m", "")
        items = ["x", "y", "z"]
        self.assertEqual(run(rr.order("q", list(items))), items)

    def test_single_doc(self):
        rr = Reranker("https://x", "m", "key")
        self.assertEqual(run(rr.rank("q", ["only"])), [(0, 1.0)])

    def test_dead_endpoint_fails_open(self):
        # key set → available, but the endpoint does not exist.
        # httpx is stubbed in the test and raises → identity expected.
        rr = Reranker("https://nonexistent.invalid/rerank", "m", "key")
        out = run(rr.rank("q", ["a", "b"]))
        self.assertEqual([i for i, _ in out], [0, 1])

    def test_factory_from_pipeline_config(self):
        cfg = PipelineConfig()
        cfg.reranker_base_url = ""
        cfg.reranker_api_key = ""
        rr = build_reranker_from_pipeline_config(cfg)
        self.assertIsInstance(rr, Reranker)
        self.assertFalse(rr.available)


class TestEmbedderFailOpen(unittest.TestCase):
    def test_unavailable_returns_none(self):
        emb = Embedder("", "m", "")
        self.assertFalse(emb.available)
        self.assertIsNone(run(emb.embed(["a", "b"])))

    def test_dead_endpoint_returns_none(self):
        emb = Embedder("https://nonexistent.invalid", "m", "key")
        self.assertIsNone(run(emb.embed(["a"])))

    def test_cosine_basics(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0, places=6)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0, places=6)
        self.assertEqual(cosine([], [1]), 0.0)
        self.assertEqual(cosine([0, 0], [0, 0]), 0.0)
        self.assertAlmostEqual(cosine([1, 1], [1, 1]), 1.0, places=6)


class TestSearchScopeClassifier(unittest.TestCase):
    def test_fallback_keeps_both_passes(self):
        """Empty LLM answer ⇒ fail-open: both additional passes active."""
        from tests._helpers import MockLLM
        llm = MockLLM()  # empty queue → default ""
        scope, call = run(classify_search_scope("irgendwas", "", llm))
        self.assertTrue(scope.time_sensitive)
        self.assertTrue(scope.academic)
        self.assertTrue(scope.fallback_used)

    def test_classifies_not_timesensitive_not_academic(self):
        from tests._helpers import MockLLM
        llm = MockLLM()
        llm.queue_json({
            "time_sensitive": False, "academic": False,
            "confidence": 0.9, "reasoning": "zeitlose Sachfrage",
        })
        scope, _ = run(classify_search_scope(
            "Was ist ein Mutex?", "", llm))
        self.assertFalse(scope.time_sensitive)
        self.assertFalse(scope.academic)
        self.assertFalse(scope.fallback_used)

    def test_classifies_timesensitive(self):
        from tests._helpers import MockLLM
        llm = MockLLM()
        llm.queue_json({
            "time_sensitive": True, "academic": False,
            "confidence": 0.95, "reasoning": "aktuelle Preise",
        })
        scope, _ = run(classify_search_scope(
            "aktueller Preis der DGX B300", "", llm))
        self.assertTrue(scope.time_sensitive)
        self.assertFalse(scope.academic)

    def test_language_whitelist_parsed(self):
        from tests._helpers import MockLLM
        llm = MockLLM()
        llm.queue_json({
            "time_sensitive": False, "academic": True,
            "languages": ["de", "en", "zh"], "recency": "",
            "confidence": 0.9, "reasoning": "KI-Thema",
        })
        scope, _ = run(classify_search_scope("KI-Modelle", "", llm))
        self.assertEqual(scope.languages, ["de", "en", "zh"])

    def test_german_only_topic(self):
        from tests._helpers import MockLLM
        llm = MockLLM()
        llm.queue_json({
            "time_sensitive": False, "academic": False,
            "languages": ["de"], "confidence": 0.92,
            "reasoning": "rein deutsches Verwaltungsthema",
        })
        scope, _ = run(classify_search_scope(
            "Promotionsordnung der Beispiel-Uni", "", llm))
        self.assertEqual(scope.languages, ["de"])

    def test_recency_validated(self):
        from tests._helpers import MockLLM
        llm = MockLLM()
        llm.queue_json({
            "time_sensitive": True, "academic": False,
            "recency": "week", "languages": [],
            "confidence": 0.9, "reasoning": "Eilmeldung",
        })
        scope, _ = run(classify_search_scope("news heute", "", llm))
        self.assertEqual(scope.recency, "week")

    def test_bogus_recency_falls_back_to_empty(self):
        from tests._helpers import MockLLM
        llm = MockLLM()
        llm.queue_json({
            "time_sensitive": True, "academic": False,
            "recency": "gestern", "languages": ["xx", "", "deu"],
            "confidence": 0.5, "reasoning": "x",
        })
        scope, _ = run(classify_search_scope("x", "", llm))
        self.assertEqual(scope.recency, "")          # invalid → ""
        # The language filter only discards empty/too long/non-alphabetic
        # codes. "xx" is structurally a valid 2-character code and stays;
        # "" is discarded.
        self.assertEqual(scope.languages, ["xx", "deu"])

    def test_fallback_languages_empty_means_no_restriction(self):
        from tests._helpers import MockLLM
        scope, _ = run(classify_search_scope("x", "", MockLLM()))
        self.assertTrue(scope.fallback_used)
        self.assertEqual(scope.languages, [])


class TestWiringRegression(unittest.TestCase):
    """Structural regression check of the wiring."""

    def setUp(self):
        self.orch = open(
            os.path.join(REPO, "src/pipeline/orchestrator.py")
        ).read()

    def test_reranker_wired_into_prefetch(self):
        self.assertIn("build_reranker_from_pipeline_config", self.orch)
        self.assertIn("reranked", self.orch)

    def test_embedder_wired_into_dedup(self):
        self.assertIn("build_embedder_from_pipeline_config", self.orch)
        self.assertIn("extract_dedup_threshold", self.orch)

    def test_search_scope_gates_fanout(self):
        self.assertIn("classify_search_scope", self.orch)
        self.assertIn("_do_recent", self.orch)
        self.assertIn("_do_science", self.orch)
        # Additional passes are gated to round 0 and by the scope.
        # The recent pass is a single line, the science pass is bracketed
        # over several lines — hence check the substrings that really occur.
        self.assertIn("round_num == 0 and _do_recent:", self.orch)
        self.assertIn("and _do_science):", self.orch)
        # language allow list wired
        self.assertIn("_effective_langs", self.orch)
        self.assertIn("_scope_langs", self.orch)
        # tunable web-search cap
        self.assertIn("max_web_searches", self.orch)

    def test_config_has_new_fields(self):
        cfg = PipelineConfig.from_env()
        for f in ("reranker_base_url", "reranker_model",
                  "reranker_api_key", "embedder_base_url",
                  "embedder_model", "embedder_api_key",
                  "extract_dedup_threshold"):
            self.assertTrue(hasattr(cfg, f), f"PipelineConfig.{f} missing")
        self.assertEqual(cfg.extract_dedup_threshold, 0.93)

    def test_map_call_disables_thinking(self):
        """The map call must have a budget for reasoning: enable_thinking=False +
        a retry before the raw-extract fallback."""
        src = self.orch
        # The map call must disable thinking
        self.assertIn("enable_thinking=False", src)
        # There must be a retry path before raw extracts are used
        # (LLM gateways return transient 504s)
        self.assertIn("one retry", src)
        self.assertIn("failed permanently", src)
        # token budget raised (from 2048)
        self.assertIn("max_tokens=4096", src)

    def test_link_following_has_relevance_gate(self):
        """Link following: rerank against the research question before
        fetching (keeps dictionary sites and the like out)."""
        src = self.orch
        self.assertIn("relevance_query", src)
        self.assertIn("Relevance gate", src)
        self.assertIn("_link_rel_q", src)
        # other_primary (web-search sources) MUST be gated
        self.assertIn("relevance_query=_link_rel_q", src)


if __name__ == "__main__":
    unittest.main()
