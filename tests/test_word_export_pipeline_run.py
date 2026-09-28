"""
Word export: pipeline-run appendix (appendix D).

For full traceability the pipeline run belongs in the Word report too. The
appendix is built from the UI renderer `render_pipeline_run` + the
Markdown→Word converter (a single source of truth). FAIL-OPEN: a render
error must never break the whole export.
"""

import os
import re
import unittest
import zipfile

import tests.conftest  # noqa: F401

from src.pipeline.models import (
    HarvestContext,
    OutputSchema,
    ResearchPlan,
    ResearchQuestion,
)
from src.pipeline.classifiers.base import ClassifierCall
from src.exporters.word_exporter import export_research_to_word


def _docx_text(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    return re.sub("<[^>]+>", "", xml)


def _make_ctx() -> HarvestContext:
    ctx = HarvestContext(query="Welche TTS-Modelle eignen sich für vLLM?")
    ctx.output_schema = OutputSchema(title="TTS Recherche")
    ctx.research_plan = ResearchPlan(
        summary="Vergleich TTS",
        questions=[ResearchQuestion(id="F1", question="Welche Modelle?")],
    )
    ctx.final_report = "# TTS Recherche\n\nHauptbericht.\n\n## Fazit\nFish-Speech."
    ctx.map_answers = [{
        "question_id": "F1", "question": "Welche Modelle?",
        "answer": "Fish-Speech, F5-TTS.", "n_extracts": 3,
    }]
    cc = ClassifierCall(
        name="search_scope", prompt_hash="h", input_summary={"q": "tts"},
    )
    cc.output = {
        "time_sensitive": True, "academic": True,
        "languages": ["de", "en"], "recency": "month", "confidence": 0.95,
    }
    ctx.classifier_calls = [cc]
    return ctx


class TestPipelineRunInWordExport(unittest.TestCase):

    def test_appendix_present_and_renumbered(self):
        fp = export_research_to_word(
            _make_ctx(), filename="/tmp/_t_pl_export.docx")
        self.assertTrue(fp and os.path.exists(fp))
        txt = _docx_text(fp)
        # New appendix D
        self.assertIn("Appendix D: Pipeline run", txt)
        # Metadata correctly moved to E (no longer D)
        self.assertIn("Appendix E: Metadata", txt)
        self.assertNotIn("Appendix D: Metadata", txt)
        # Main report still included
        self.assertIn("Hauptbericht", txt)
        # Classifier decision visible
        self.assertIn("search_scope", txt)

    def test_export_survives_unrenderable_ctx(self):
        """Fail-open: even an almost empty ctx must not abort the export —
        the pipeline-run appendix is then just short."""
        ctx = HarvestContext(query="x")
        ctx.output_schema = OutputSchema(title="Leer")
        ctx.final_report = "# Leer\n\nNichts."
        fp = export_research_to_word(
            ctx, filename="/tmp/_t_pl_export_empty.docx")
        self.assertTrue(fp and os.path.exists(fp))
        txt = _docx_text(fp)
        # The appendix heading exists anyway
        self.assertIn("Appendix D: Pipeline run", txt)


if __name__ == "__main__":
    unittest.main()


def test_word_export_follows_output_language(monkeypatch, tmp_path):
    """With German as the output language the appendices are German."""
    from src import output_language
    from src.exporters.word_exporter import export_research_to_word
    from src.pipeline.models import HarvestContext
    monkeypatch.setenv("OUTPUT_LANGUAGES", "en,de")
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    output_language.reload()
    try:
        ctx = HarvestContext(query="Testanfrage")
        ctx.output_language = "de"
        ctx.final_report = "# Bericht\n\nInhalt."
        path = export_research_to_word(ctx)
        from docx import Document
        txt = "\n".join(p.text for p in Document(path).paragraphs)
        assert "Recherche-Bericht" in txt and "Erstellt am" in txt
        assert "Anhang E: Metadaten" in txt
        assert "Hinweise zur Erstellung" in txt
    finally:
        monkeypatch.delenv("OUTPUT_LANGUAGES")
        output_language.reload()
