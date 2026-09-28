"""
Regression: PDF detection in the web scraper.

`resp.text` on PDF bytes yields decoded binary garbage; trafilatura
discards it and the simple-strip fallback returns ~500k characters of
junk, which would end up in the reranker, in the off-topic filter
(content[:500] = PDF header → discarded as off_topic) and in the harvest —
relevant papers would be lost.

PDFs are therefore recognised by Content-Type/%PDF magic and extracted
with pdfminer; on error a SHORT marker is returned instead of binary junk.

These tests check the core behaviour: never binary garbage, always short
and clean on failure.
"""

import asyncio
import unittest

import tests.conftest  # noqa: F401

from src.connectors.web_scraper import WebScraperConnector


def run(coro):
    return asyncio.run(coro)


class TestPdfExtractionFailClean(unittest.TestCase):

    def test_broken_pdf_yields_short_marker_not_garbage(self):
        out = run(WebScraperConnector._extract_pdf_text(
            b"%PDF-1.4 hopelessly broken", "http://x/a.pdf"))
        self.assertTrue(out.startswith("["))
        self.assertIn("a.pdf", out)
        # Crucial: SHORT, not 500k of binary junk
        self.assertLess(len(out), 200)

    def test_empty_bytes_fail_clean(self):
        out = run(WebScraperConnector._extract_pdf_text(
            b"", "http://x/empty.pdf"))
        self.assertTrue(out.startswith("["))
        self.assertLess(len(out), 200)

    def test_no_text_layer_marked_not_garbage(self):
        # A structurally valid but text-less PDF → a clear marker,
        # NO binary junk (otherwise the off-topic filter would again only
        # see the PDF header).
        minimal = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF"
        )
        out = run(WebScraperConnector._extract_pdf_text(
            minimal, "http://x/scan.pdf"))
        self.assertTrue(out.startswith("["))
        self.assertIn("scan.pdf", out)
        self.assertLess(len(out), 200)

    def test_never_returns_raw_binary(self):
        """Core contract: the result NEVER contains the raw PDF bytes."""
        blob = b"%PDF-1.4\n" + b"\x00\x01\x02BINARY" * 5000
        out = run(WebScraperConnector._extract_pdf_text(
            blob, "http://x/big.pdf"))
        self.assertNotIn("BINARY", out)
        self.assertLess(len(out), 500)  # never ~500k


if __name__ == "__main__":
    unittest.main()
