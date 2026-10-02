"""
Last result in the browser (gr.BrowserState): snapshot and restore.

Only finished text is stored: the tabs as shown, the pipeline-run view
and the report as exported. The snapshot is written once per result,
stays below the size limit, and a restored result exports to Markdown
unchanged and to Word with sources and extracts as appendices.
"""

import json
import os
import unittest
from unittest.mock import patch

import gradio as gr

import tests.conftest  # noqa: F401  (httpx/openai stubs)
from src.pipeline.models import HarvestContext, OutputSchema
from src.ui import gradio_app as ga

SKIP = gr.skip()


def _ctx(report="# Bericht\n\nInhalt.", format_type="structured_report"):
    ctx = HarvestContext(query="Sind Vermögenssteuern sinnvoll?",
                         output_language="de")
    ctx.final_report = report
    ctx.output_schema = OutputSchema(title="Vermögenssteuern",
                                     format_type=format_type, style="sachlich")
    ctx.finished_at = ctx.started_at
    ctx.status = "done"
    return ctx


def _state(ctx=None):
    st = ga.AppState()
    st.current_research = ctx
    return st


TABS = dict(report="# Bericht\n\nInhalt.", sources="### 🔗 Quellen (2)",
            progress="Runde 1 fertig", extracts="### 📝 Extrakte (5)")


def _save(st, **tabs):
    t = {**TABS, **tabs}
    with patch.object(ga, "_render_pipeline_run_text", return_value="## DAG"):
        return ga.save_result_to_browser(st, t["report"], t["sources"],
                                         t["progress"], t["extracts"])


class _NoReadyState(unittest.TestCase):
    def setUp(self):
        p = patch.object(ga, "_get_ready_state", side_effect=lambda s: s)
        p.start()
        self.addCleanup(p.stop)


class TestSaveResult(_NoReadyState):
    def test_snapshot_is_json_with_all_tabs(self):
        st = _state(_ctx())
        state, snap = _save(st)
        self.assertIs(state, st)
        json.dumps(snap)  # must not raise
        self.assertEqual(snap["version"], 1)
        self.assertEqual(snap["report"], TABS["report"])
        self.assertEqual(snap["sources"], TABS["sources"])
        self.assertEqual(snap["progress"], TABS["progress"])
        self.assertEqual(snap["extracts"], TABS["extracts"])
        self.assertEqual(snap["pipeline_run"], "## DAG")
        self.assertEqual(snap["title"], "Vermögenssteuern")
        # export text equals the shown report → stored only once
        self.assertIsNone(snap["export_md"])

    def test_export_text_kept_when_it_differs(self):
        st = _state(_ctx(report="# Bericht\n\nInhalt.\n\n---\n\nFilter-Banner"))
        _, snap = _save(st)
        self.assertEqual(snap["export_md"], "# Bericht\n\nInhalt.\n\n---\n\nFilter-Banner")

    def test_written_once_per_result(self):
        st = _state(_ctx())
        _save(st)
        self.assertEqual(_save(st), (SKIP, SKIP))

    def test_nothing_without_finished_result(self):
        self.assertEqual(_save(_state()), (SKIP, SKIP))
        st = _state(_ctx())
        st.research_running = True
        self.assertEqual(_save(st), (SKIP, SKIP))

    def test_bounded_and_report_shortened_last(self):
        big = "Zeile mit Text äöü 🔬\n" * 40_000  # about 1 MB
        st = _state(_ctx())
        _, snap = _save(st, extracts=big, sources=big)
        self.assertLessEqual(ga._json_size(snap), ga.RESULT_STORE_MAX_BYTES)
        self.assertEqual(snap["truncated"][:2], ["pipeline_run", "extracts"])
        self.assertNotIn("report", snap["truncated"])
        self.assertEqual(snap["report"], TABS["report"])
        self.assertIn("…", snap["sources"] + snap["extracts"])

    def test_bounded_even_with_huge_report(self):
        big = "x" * 600_000
        st = _state(_ctx(report=big + "!"))
        _, snap = _save(st, report=big, extracts=big)
        self.assertLessEqual(ga._json_size(snap), ga.RESULT_STORE_MAX_BYTES)
        self.assertIn("report", snap["truncated"])


class TestRestoreResult(_NoReadyState):
    def _round_trip(self, ctx, **tabs):
        _, snap = _save(_state(ctx), **tabs)
        snap = json.loads(json.dumps(snap))  # as it comes back from the browser
        return ga.restore_result_from_browser(snap, _state())

    def test_reopens_result_area_marked_as_restored(self):
        state, panel, report, sources, progress, extracts, run = \
            self._round_trip(_ctx())
        self.assertEqual(panel, gr.update(visible=True))
        self.assertTrue(state.result_panel_visible)
        self.assertIn("Restored result", report.split("\n")[0])
        self.assertTrue(report.endswith(TABS["report"]))
        self.assertEqual((sources, progress, extracts),
                         (TABS["sources"], TABS["progress"], TABS["extracts"]))
        self.assertEqual(run, "## DAG")
        self.assertEqual(ga._restored_snapshot(state)["pipeline_run"], "## DAG")

    def test_markdown_export_unchanged(self):
        ctx = _ctx(report="# Bericht\n\nText\n\n---\n\nBanner")
        state, *_ = self._round_trip(ctx)
        restored = state.current_research
        self.assertEqual(restored.final_report, ctx.final_report)
        self.assertEqual(restored.output_schema.title, "Vermögenssteuern")
        self.assertEqual(restored.output_language, "de")
        with patch.object(ga.gr, "Info"):
            path = ga.export_markdown(state)
        self.addCleanup(os.remove, path)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), ctx.final_report)

    def test_word_context_appends_sources_and_extracts(self):
        state, *_ = self._round_trip(_ctx())
        word_ctx = ga._word_export_context(state)
        self.assertIsNot(word_ctx, state.current_research)
        self.assertEqual(
            word_ctx.final_report,
            "\n\n---\n\n".join([TABS["report"], TABS["sources"], TABS["extracts"]]))
        # the Markdown export keeps the plain report
        self.assertEqual(state.current_research.final_report, TABS["report"])

    def test_word_context_of_reference_check_unchanged(self):
        state, *_ = self._round_trip(_ctx(format_type="literature_check"))
        self.assertIs(ga._word_export_context(state), state.current_research)

    def test_word_context_of_live_result_unchanged(self):
        st = _state(_ctx())
        self.assertIs(ga._word_export_context(st), st.current_research)
        _save(st)
        self.assertIs(ga._word_export_context(st), st.current_research)

    def test_word_export_produces_file(self):
        try:
            import docx  # noqa: F401
        except ImportError:
            self.skipTest("python-docx not installed")
        state, *_ = self._round_trip(_ctx())
        with patch.object(ga.gr, "Info"), patch.object(ga.gr, "Warning") as warn:
            path = ga.export_word(state)
        self.assertTrue(path and os.path.exists(path), warn.call_args)
        self.addCleanup(os.remove, path)

    def test_bibtex_export_explains_itself(self):
        state, *_ = self._round_trip(_ctx(format_type="literature_check"))
        with patch.object(ga.gr, "Warning") as warn:
            self.assertIsNone(ga.export_bibtex(state))
        self.assertIn("restored", warn.call_args.args[0])

    def test_empty_or_foreign_data_changes_nothing(self):
        for stored in (None, {}, "kaputt", {"version": 99, "report": "x"},
                       {"version": 1, "report": "  "}, {"version": 1}):
            st = _state()
            self.assertEqual(ga.restore_result_from_browser(stored, st),
                             (SKIP,) * 7)
            self.assertIsNone(st.current_research)

    def test_malformed_fields_are_tolerated(self):
        stored = {"version": 1, "report": "# R", "sources": 5,
                  "export_md": ["x"], "saved_at": "gestern",
                  "started_at": None, "truncated": "extracts"}
        state, _, report, sources, *_ = ga.restore_result_from_browser(stored, _state())
        self.assertTrue(report.endswith("# R"))
        self.assertIn("gestern", report)
        self.assertEqual(sources, "")
        self.assertEqual(state.current_research.final_report, "# R")
        self.assertGreaterEqual(state.current_research.duration_seconds, 0)

    def test_new_run_replaces_restored_result(self):
        state, *_ = self._round_trip(_ctx())
        new_ctx = _ctx(report="# Neu")
        state.current_research = new_ctx
        # not yet saved: the pipeline view must render the new run
        self.assertIsNone(ga._restored_snapshot(state))
        _, snap = _save(state, report="# Neu")
        self.assertEqual(snap["report"], "# Neu")
        self.assertIsNone(state.restored_result)
        self.assertIs(ga._word_export_context(state), new_ctx)


class TestNewChatDiscardsResult(_NoReadyState):
    def test_discard_clears_store_and_is_not_saved_again(self):
        st = _state(_ctx())
        _save(st)
        state, stored = ga.discard_result_in_browser(st)
        # truthy, otherwise the browser keeps the old value
        self.assertEqual(stored, {})
        self.assertEqual(ga.restore_result_from_browser(stored, _state()),
                         (SKIP,) * 7)
        # the next research chain without a new result writes nothing
        self.assertEqual(_save(state), (SKIP, SKIP))

    def test_discard_before_first_save(self):
        st = _state(_ctx())
        ga.discard_result_in_browser(st)
        self.assertEqual(_save(st), (SKIP, SKIP))


if __name__ == "__main__":
    unittest.main()
