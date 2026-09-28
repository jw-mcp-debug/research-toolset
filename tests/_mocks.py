"""Shared test doubles."""



class MockSubClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0


class SmartMockLLM:
    """Mock DualLLMClient that returns matching answers depending on the prompt content."""

    def __init__(self):
        self.primary = MockSubClient("mock-primary")
        self.harvest = MockSubClient("mock-harvest")

    def _accounting(self, client_obj):
        client_obj.total_requests += 1
        client_obj.total_prompt_tokens += 200
        client_obj.total_completion_tokens += 100

    async def primary_complete(self, messages, max_tokens=None,
                                enable_thinking=None) -> str:
        self._accounting(self.primary)
        prompt = messages[0]["content"]
        return self._answer_text(prompt)

    async def primary_complete_json(self, messages, max_tokens=None) -> dict:
        self._accounting(self.primary)
        prompt = messages[0]["content"]
        return self._answer_json(prompt)

    async def harvest_complete(self, messages, max_tokens=None) -> str:
        self._accounting(self.harvest)
        prompt = messages[0]["content"]
        return self._answer_text(prompt)

    def get_usage_stats(self) -> dict:
        return {
            "primary": {
                "model": self.primary.model_name,
                "requests": self.primary.total_requests,
                "prompt_tokens": self.primary.total_prompt_tokens,
                "completion_tokens": self.primary.total_completion_tokens,
                "total_tokens": (self.primary.total_prompt_tokens
                                 + self.primary.total_completion_tokens),
            },
            "harvest": {
                "model": self.harvest.model_name,
                "requests": self.harvest.total_requests,
                "prompt_tokens": self.harvest.total_prompt_tokens,
                "completion_tokens": self.harvest.total_completion_tokens,
                "total_tokens": (self.harvest.total_prompt_tokens
                                 + self.harvest.total_completion_tokens),
            },
        }

    # ─── Answer heuristics ───

    def _answer_json(self, prompt: str) -> dict:
        """Return JSON structures matching the planning prompt."""
        # PEER REVIEW Planning
        if "Gutachten" in prompt and "sections" in prompt:
            return {
                "sections": ["Abstract", "Einleitung", "Methoden", "Ergebnisse"],
                "criteria": ["Originalität", "Methodik", "Klarheit", "Beleglage"],
                "paper_type": "empirisch",
                "focus_note": "Fokus auf Methodik",
            }
        # PEER REVIEW recommendation (also JSON)
        if "Wähle EINE Empfehlung" in prompt:
            return {
                "recommendation": "Major Revisions",
                "summary": "Solide Grundlage, methodische Mängel.",
                "must_address": ["Stichprobengröße", "Statistik"],
            }
        # PEER REVIEW criterion (comes as text, not JSON — dispatching)
        # DECISION ANALYSIS planning
        if "Entscheidungsanalyse" in prompt and "criteria" in prompt:
            return {
                "criteria": [
                    {"name": "Kosten", "weight": 30, "rationale": "..."},
                    {"name": "Nutzen", "weight": 40, "rationale": "..."},
                    {"name": "Risiko", "weight": 30, "rationale": "..."},
                ],
                "blind_spots": ["Langzeit", "Stakeholder X"],
            }
        # DECISION evaluation
        if "AUSSCHLIESSLICH dieses eine Option-Kriterium" in prompt:
            return {
                "score": 4,
                "reasoning": "Mock-Begründung",
                "uncertainty": "low",
                "key_evidence": ["Punkt 1"],
            }
        # DECISION Aggregation
        if "Aggregiere die Einzelbewertungen" in prompt:
            return {
                "options": [
                    {"name": "Option A", "weighted_score": 3.85,
                     "characterization": "Ausgewogen",
                     "best_criteria": ["Kosten"], "worst_criteria": ["Risiko"]},
                ],
                "ranking": ["Option A", "Option B"],
            }
        # RESEARCH DESIGN Planning
        if "Forschungsdesign" in prompt and "candidate_methods" in prompt:
            return {
                "candidate_methods": [
                    {"name": "Survey", "fit": "hoch", "rationale": "..."},
                    {"name": "Interviews", "fit": "mittel", "rationale": "..."},
                ],
                "key_decisions": ["Sampling", "Operationalisierung"],
                "discipline_traditions": ["Tradition A"],
            }
        # RESEARCH DESIGN Method comparison
        if "fit_score" in prompt and "method" in prompt:
            return {
                "method": "Survey",
                "fit_score": 4,
                "strengths": ["Skalierbar", "Standardisiert"],
                "weaknesses": ["Oberflächlich"],
                "preconditions": ["Validiertes Instrument"],
                "pitfalls": ["Antwort-Bias"],
                "summary": "Passt gut",
            }
        # RESEARCH DESIGN Method recommendation
        if "Methode wird empfohlen" in prompt:
            return {
                "primary_method": "Survey + qualitative Interviews",
                "rationale": "Mixed-Methods passt zur Frage",
                "alternative": "Reine Survey",
                "alternative_rationale": "...",
                "excluded": ["Experiment (nicht praktikabel)"],
            }
        # GRANT PROPOSAL Planning
        if "Drittmittelantrag" in prompt and "work_packages" in prompt:
            return {
                "sections": ["Stand", "Ziele", "Methodik", "AP", "Verwertung"],
                "main_goals": ["Ziel 1", "Ziel 2", "Ziel 3"],
                "work_packages": [
                    {"id": "WP1", "title": "Vorbereitung", "months": "M1-M6"},
                    {"id": "WP2", "title": "Durchführung", "months": "M7-M24"},
                    {"id": "WP3", "title": "Auswertung", "months": "M25-M36"},
                ],
                "critical_points": ["Methodik", "Budget"],
            }
        # GRANT PROPOSAL Consistency check
        if "consistency_status" in prompt:
            return {
                "consistency_status": "alle Prüfungen bestanden",
                "issues": [],
                "recommendations": ["Vor Einreichung Korrektur lesen"],
            }
        # LIT REVIEW Planning
        if "Literature Review" in prompt or "extraction_fields" in prompt:
            return {
                "operationalized_question": "Mock-Operationalisierung",
                "sub_questions": ["Sub 1", "Sub 2"],
                "extraction_fields": [
                    "Studientyp", "Stichprobe", "Methode", "Hauptergebnis",
                ],
                "quality_criteria": [
                    "Methodische Transparenz", "Statistische Solidität",
                ],
                "estimated_inclusion_rate": "70%",
            }
        # LIT REVIEW Screening
        if "decision" in prompt and "include" in prompt:
            return {
                "decision": "include",
                "reason": "Relevante Methodik",
                "relevance_score": 4,
                "matched_criteria": ["Methodik passt"],
                "missing_criteria": [],
            }
        # LIT REVIEW Extraction
        if "extracted" in prompt:
            return {
                "title": "Mock Paper Title",
                "extracted": {
                    "Studientyp": "Querschnitt",
                    "Stichprobe": "n=200",
                    "Methode": "Survey",
                    "Hauptergebnis": "Effekt nachgewiesen",
                },
                "key_quote": "Ein zentraler Befund.",
                "relation_to_question": "Beantwortet Sub-Frage 1",
            }
        # LIT REVIEW Quality
        if "criteria_scores" in prompt:
            return {
                "overall_quality": "stark",
                "criteria_scores": {
                    "Methodische Transparenz": {"score": 4, "rationale": "..."},
                    "Statistische Solidität": {"score": 4, "rationale": "..."},
                },
                "concerns": [],
            }
        # LIT REVIEW Clustering
        if "clusters" in prompt:
            return {
                "clusters": [
                    {
                        "name": "Cluster A: Methodische Studien",
                        "theme": "Theme A",
                        "papers": ["P01", "P02"],
                        "consensus": "Konsens",
                        "tensions": "Keine",
                    },
                ],
            }
        # EXPLAINER Planning
        if "Tiefenerklärung" in prompt or "core_concepts" in prompt:
            return {
                "audience_summary": "Mock audience",
                "core_concepts": ["Konzept 1", "Konzept 2", "Konzept 3"],
                "scope_note": "Scope",
            }
        return {}

    def _answer_text(self, prompt: str) -> str:
        """Generic text-output mock."""
        # Recognise synthesis prompts → longer text
        if "Erstelle das finale" in prompt or "Erstelle den finalen" in prompt:
            return (
                "# Mock Finaler Bericht\n\n"
                "## Zusammenfassung\nMock-Inhalt mit ausreichend Text.\n\n"
                "## Hauptteil\nDetaillierte Mock-Erklärungen.\n\n"
                "## Fazit\nMock-Schluss.\n"
            )
        # Standard output
        return "Mock-Text-Output (mittlere Länge zur Demonstration)."
