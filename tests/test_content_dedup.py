"""
Tests for `src.pipeline.content_dedup`.

  - sources with identical content are removed except for one.
  - with mirror subdomains the main domain wins.
  - sources with too little content (< 200 characters) stay unchanged.
  - the original order is kept among the surviving sources.
  - URLs in the content do not disturb duplicate detection.
"""

import unittest
from dataclasses import dataclass

from src.pipeline.content_dedup import dedupe_sources_by_content


@dataclass
class _Src:
    """Minimal duck type for SourceDocument (only url + content)."""
    url: str
    content: str


class TestDedupeSourcesByContent(unittest.TestCase):

    def test_no_duplicates_keeps_all(self):
        sources = [
            _Src("https://a.com", "Content A " * 100),
            _Src("https://b.com", "Content B " * 100),
            _Src("https://c.com", "Content C " * 100),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        self.assertEqual(len(kept), 3)
        self.assertEqual(removed, 0)

    def test_identical_content_collapses(self):
        same_text = "Identischer Inhalt " * 100
        sources = [
            _Src("https://a.com", same_text),
            _Src("https://b.com", same_text),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed, 1)

    def test_mirror_subdomain_main_wins(self):
        """The main domain is preferred over a subdomain."""
        same_text = "Uni-Inhalt über Forschung " * 100
        sources = [
            _Src("https://cms.example.edu/page", same_text),
            _Src("https://example.edu/page", same_text),
            _Src("https://committees.example.edu/page", same_text),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        self.assertEqual(len(kept), 1)
        # The shortest URL (= main domain) wins
        self.assertEqual(kept[0].url, "https://example.edu/page")
        self.assertEqual(removed, 2)

    def test_short_content_not_deduped(self):
        """Short content (< 200 characters) is not de-duplicated.

        Safe default: two short stub pages could have the same text by
        chance; we do not want to collapse them by mistake.
        """
        short = "Kurz."
        sources = [
            _Src("https://a.com", short),
            _Src("https://b.com", short),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        # both stay
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, 0)

    def test_urls_in_content_dont_prevent_dedup(self):
        """URLs in the content differ between mirrors — they must not prevent
        duplicate detection."""
        # Both have identical main content, but different URLs
        content_a = (
            "Forschung über KI. Mehr unter https://cms.example.edu/details "
            "und im Hauptartikel. " * 50
        )
        content_b = (
            "Forschung über KI. Mehr unter https://committees.example.edu/details "
            "und im Hauptartikel. " * 50
        )
        sources = [
            _Src("https://cms.example.edu/page", content_a),
            _Src("https://committees.example.edu/page", content_b),
        ]
        kept, removed = dedupe_sources_by_content(sources)

        # Identified as a duplicate (URL stripped from the content)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed, 1)

    def test_whitespace_normalized(self):
        """Different whitespace must not count as a difference."""
        a = "Inhalt  mit   viel   whitespace " * 50
        b = "Inhalt mit viel whitespace " * 50
        sources = [
            _Src("https://a.com", a),
            _Src("https://b.com", b),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        self.assertEqual(len(kept), 1)

    def test_case_insensitive(self):
        a = "Forschung Über KI " * 50
        b = "FORSCHUNG ÜBER KI " * 50
        sources = [
            _Src("https://a.com", a),
            _Src("https://b.com", b),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        self.assertEqual(len(kept), 1)

    def test_preserves_original_order(self):
        """Surviving sources keep their relative original order."""
        unique = "Einzigartiger Inhalt " * 50
        same = "Doppelter Inhalt " * 50
        sources = [
            _Src("https://x.com", unique + " 1"),
            _Src("https://aaa.com", same),
            _Src("https://y.com", unique + " 2"),
            _Src("https://b.com", same),  # mirror of aaa
            _Src("https://z.com", unique + " 3"),
        ]
        kept, removed = dedupe_sources_by_content(sources)
        # 4 sources expected (3 unique + 1 of the same mirrors)
        self.assertEqual(len(kept), 4)
        self.assertEqual(removed, 1)

        # the order of the unique sources is kept
        urls = [s.url for s in kept]
        self.assertEqual(urls.index("https://x.com"), 0)
        self.assertLess(urls.index("https://x.com"), urls.index("https://y.com"))
        self.assertLess(urls.index("https://y.com"), urls.index("https://z.com"))

    def test_empty_input(self):
        kept, removed = dedupe_sources_by_content([])
        self.assertEqual(kept, [])
        self.assertEqual(removed, 0)

    def test_none_content_handled(self):
        """A source with None content is not treated as a duplicate."""
        sources = [
            _Src("https://a.com", None),
            _Src("https://b.com", "Content B " * 50),
        ]
        # should not crash
        kept, removed = dedupe_sources_by_content(sources)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
