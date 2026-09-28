"""
Tests for the diagnosis banner appended to the report.

Acceptance: with `is_problematic=True` a Markdown block "⚠️ Hinweis zum
Recherche-Verlauf" with user_message and remediation is appended to the
report. With `is_successful=True` the report stays unchanged.

These tests check the banner behaviour in isolation from the pipeline run:
they simulate the ctx.final_diagnosis structure and check the string
manipulation.
"""

import unittest


def _append_diagnosis_banner(report: str, final_diagnosis: dict | None) -> str:
    """Mirrors the banner logic of the pipeline.

    NOTE: this is a copy, not the real code (DiagnosisBannerNode in
    src/pipeline/dag_nodes.py). If the banner code changes, this function
    has to be kept in sync by hand.
    """
    if final_diagnosis and final_diagnosis.get("is_problematic"):
        d = final_diagnosis
        banner = (
            "\n\n---\n\n"
            "## ⚠️ Hinweis zum Recherche-Verlauf\n\n"
            f"<!-- diagnosis: {d['diagnosis']} -->\n\n"
            f"{d.get('user_message', '').strip()}\n"
        )
        remediation = (d.get("remediation") or "").strip()
        if remediation:
            banner += f"\n**Empfehlung:** {remediation}\n"
        return report + banner
    return report


class TestDiagnosisBanner(unittest.TestCase):

    def test_no_diagnosis_no_banner(self):
        report = "## Bericht\n\nInhalt."
        result = _append_diagnosis_banner(report, None)
        self.assertEqual(result, report)

    def test_successful_diagnosis_no_banner(self):
        diag = {
            "diagnosis": "successful", "is_problematic": False,
            "is_successful": True,
            "user_message": "Erfolgreich.",
            "remediation": "",
        }
        report = "## Bericht\n\nInhalt."
        result = _append_diagnosis_banner(report, diag)
        self.assertEqual(result, report)

    def test_filter_too_strict_appends_banner(self):
        """CRITICAL: a run in which a filter discarded everything becomes visible in the report."""
        diag = {
            "diagnosis": "filter_too_strict",
            "is_problematic": True,
            "is_successful": False,
            "user_message": (
                "Die Recherche besuchte 78 Quellen, alle 528 Extrakte "
                "wurden vom Filter verworfen."
            ),
            "remediation": (
                "Bitte starten Sie die Recherche ohne Personen-"
                "Halluzinations-Filter neu."
            ),
            "confidence": 0.95,
        }
        report = "## Bericht\n\nKurzer Inhalt."

        result = _append_diagnosis_banner(report, diag)

        # the report is kept
        self.assertTrue(result.startswith("## Bericht"))
        # the banner is appended
        self.assertIn("⚠️ Hinweis zum Recherche-Verlauf", result)
        # machine-readable tag (for later UI parsing) in it
        self.assertIn("<!-- diagnosis: filter_too_strict -->", result)
        # user message in it
        self.assertIn("alle 528 Extrakte", result)
        # recommendation in it
        self.assertIn("**Empfehlung:**", result)
        self.assertIn("ohne Personen-Halluzinations-Filter", result)

    def test_diagnosis_without_remediation(self):
        diag = {
            "diagnosis": "data_scarcity",
            "is_problematic": True,
            "is_successful": False,
            "user_message": "Wenig Datenlage.",
            "remediation": "",  # empty
        }
        result = _append_diagnosis_banner("Bericht", diag)
        self.assertIn("Wenig Datenlage.", result)
        # No recommendation, no recommendation block
        self.assertNotIn("**Empfehlung:**", result)

    def test_banner_has_horizontal_rule(self):
        """The banner is visually separated from the report."""
        diag = {
            "diagnosis": "wrong_query", "is_problematic": True,
            "is_successful": False,
            "user_message": "Anfrage scheint zu eng.",
            "remediation": "Anfrage breiter formulieren.",
        }
        result = _append_diagnosis_banner("Bericht", diag)
        # a horizontal rule separates
        self.assertIn("---", result)


if __name__ == "__main__":
    unittest.main()
