"""
Tests for `src.pipeline.analysis_pipeline`.

Acceptance:
  - is_analysis_use_case recognises the right modes
  - AnalysisLayerNode runs tasks in parallel
  - tasks with failed dependencies are skipped
  - dependency-output injection via {dep:ID} and param_deps works
  - the retry logic kicks in for transient errors
  - the decomposer builds working plans per use case
  - AnalysisPipelineRunner wraps it all in a DAG
"""

import asyncio
import unittest
from dataclasses import dataclass, field

from src.pipeline.analysis_pipeline import (
    ANALYSIS_USE_CASES,
    AnalysisLayerNode,
    AnalysisPipelineRunner,
    Decomposer,
    is_analysis_use_case,
)
from src.pipeline.models import SubTask, TaskPlan, TaskResult, TaskState
from tests._helpers import async_test


@dataclass
class _MinimalCtx:
    """Minimal ctx stub with the fields analysis_pipeline touches."""
    query: str = "Erkläre Quantenmechanik"
    use_case: str = "explainer"
    task_plan: TaskPlan = None
    task_results: dict = field(default_factory=dict)
    preflight_data: dict = field(default_factory=dict)
    status: str = "running"
    error_message: str = ""
    started_at: str = ""
    finished_at: str = ""
    node_results: list = field(default_factory=list)


class _MockLLMClient:
    """Mock LLM with a fixed answer per prompt substring."""
    def __init__(self, responses: dict, fail_count: dict = None):
        # responses: dict[str_substr] -> str_response
        # fail_count: dict[str_substr] -> int (how often the call should
        #             fail before it succeeds)
        self.responses = responses
        self.fail_count = dict(fail_count or {})
        self.calls: list[str] = []

    async def harvest_complete(self, messages):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        for k, v in self.responses.items():
            if k in prompt:
                if self.fail_count.get(k, 0) > 0:
                    self.fail_count[k] -= 1
                    raise RuntimeError(f"mock failure ({self.fail_count[k]} remaining)")
                return v
        raise KeyError(f"No mock response matches the prompt: {prompt[:80]!r}")

    async def primary_complete_json(self, messages):
        # Real behaviour: returns a dict, not a string. We parse the
        # configured string answer as JSON, or return an empty dict for
        # broken answers.
        text = await self.harvest_complete(messages)
        try:
            import json as _json
            data = _json.loads(text)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}


# ───────────────────────────────────────────────────────────────────
# is_analysis_use_case
# ───────────────────────────────────────────────────────────────────


class TestIsAnalysisUseCase(unittest.TestCase):

    def test_recognized_modes(self):
        for mode in ANALYSIS_USE_CASES:
            self.assertTrue(is_analysis_use_case(mode), f"mode {mode!r} should be an analysis")

    def test_unrecognized_modes(self):
        for mode in ["web", "institution", "product_comparison", "unknown", ""]:
            self.assertFalse(is_analysis_use_case(mode), f"mode {mode!r} is NOT an analysis")


# ───────────────────────────────────────────────────────────────────
# AnalysisLayerNode — layer execution
# ───────────────────────────────────────────────────────────────────


class TestLayerExecution(unittest.TestCase):

    @async_test
    async def test_single_task_executes(self):
        task = SubTask(
            id="T1", phase="p1", description="Test",
            prompt_template="Analysiere: {topic}",
            prompt_params={"topic": "Beispiel"},
        )
        llm = _MockLLMClient(responses={"Analysiere": "ANALYSE-OUTPUT"})

        ctx = _MinimalCtx()
        node = AnalysisLayerNode(layer_idx=0, tasks=[task], llm_client=llm)
        meta = await node.run(ctx)

        self.assertEqual(meta["done"], 1)
        self.assertIn("T1", ctx.task_results)
        self.assertEqual(ctx.task_results["T1"].state, TaskState.DONE)
        self.assertEqual(ctx.task_results["T1"].output, "ANALYSE-OUTPUT")

    @async_test
    async def test_parallel_execution_in_layer(self):
        """Three tasks in one layer run in parallel."""
        tasks = [
            SubTask(id=f"T{i}", phase="p", description="x",
                    prompt_template=f"Marker_T{i}",
                    prompt_params={})
            for i in range(3)
        ]
        # every task has its own marker in the prompt
        llm = _MockLLMClient(responses={
            "Marker_T0": "out-0",
            "Marker_T1": "out-1",
            "Marker_T2": "out-2",
        })

        ctx = _MinimalCtx()
        node = AnalysisLayerNode(layer_idx=0, tasks=tasks, llm_client=llm)
        meta = await node.run(ctx)

        self.assertEqual(meta["done"], 3)
        self.assertEqual(ctx.task_results["T0"].output, "out-0")
        self.assertEqual(ctx.task_results["T1"].output, "out-1")
        self.assertEqual(ctx.task_results["T2"].output, "out-2")

    @async_test
    async def test_skipped_when_dependency_failed(self):
        """A task with a failed dependency is skipped."""
        # T1 is already marked FAILED in ctx
        ctx = _MinimalCtx(task_results={
            "T1": TaskResult(task_id="T1", state=TaskState.FAILED,
                             error="Erstes Failure"),
        })
        # T2 depends on T1
        task = SubTask(
            id="T2", phase="p", description="x",
            prompt_template="x", depends_on=["T1"],
        )
        llm = _MockLLMClient(responses={})  # must not be called

        node = AnalysisLayerNode(layer_idx=1, tasks=[task], llm_client=llm)
        meta = await node.run(ctx)

        self.assertEqual(meta["skipped"], 1)
        self.assertEqual(meta["executed"], 0)
        self.assertEqual(ctx.task_results["T2"].state, TaskState.SKIPPED)
        self.assertIn("T1", ctx.task_results["T2"].error)
        # the LLM was NOT called
        self.assertEqual(len(llm.calls), 0)


# ───────────────────────────────────────────────────────────────────
# Dependency-Output-Injection
# ───────────────────────────────────────────────────────────────────


class TestDepInjection(unittest.TestCase):

    @async_test
    async def test_dep_marker_substituted(self):
        """{dep:T1} is replaced by T1.output."""
        ctx = _MinimalCtx(task_results={
            "T1": TaskResult(task_id="T1", state=TaskState.DONE,
                             output="ERSTER_OUTPUT"),
        })
        task = SubTask(
            id="T2", phase="p", description="x",
            prompt_template="Aufbauend auf: {dep:T1} weitermachen",
            depends_on=["T1"],
        )
        llm = _MockLLMClient(responses={"ERSTER_OUTPUT": "T2-OUT"})

        node = AnalysisLayerNode(layer_idx=1, tasks=[task], llm_client=llm)
        await node.run(ctx)

        # the LLM got the substituted prompt
        self.assertIn("ERSTER_OUTPUT", llm.calls[0])
        self.assertNotIn("{dep:T1}", llm.calls[0])

    @async_test
    async def test_param_dep_substituted(self):
        """param_deps fills template variables from dependency outputs."""
        ctx = _MinimalCtx(task_results={
            "T1": TaskResult(task_id="T1", state=TaskState.DONE,
                             output="VORHERIGE_ANALYSE"),
        })
        task = SubTask(
            id="T2", phase="p", description="x",
            prompt_template="Bewerte: {analyse}",
            param_deps={"analyse": "T1"},
            depends_on=["T1"],
        )
        llm = _MockLLMClient(responses={"VORHERIGE_ANALYSE": "T2-OUT"})

        node = AnalysisLayerNode(layer_idx=1, tasks=[task], llm_client=llm)
        await node.run(ctx)

        self.assertIn("VORHERIGE_ANALYSE", llm.calls[0])

    @async_test
    async def test_concat_param_dep_joins_outputs(self):
        """concat_param_deps concatenates several dependency outputs."""
        ctx = _MinimalCtx(task_results={
            "T1": TaskResult(task_id="T1", state=TaskState.DONE,
                             output="OUTPUT_EINS"),
            "T2": TaskResult(task_id="T2", state=TaskState.DONE,
                             output="OUTPUT_ZWEI"),
        })
        task = SubTask(
            id="T3", phase="p", description="x",
            prompt_template="Synthese aus: {teile}",
            concat_param_deps={"teile": ["T1", "T2"]},
            depends_on=["T1", "T2"],
        )
        llm = _MockLLMClient(responses={"Synthese aus": "T3-OUT"})

        node = AnalysisLayerNode(layer_idx=1, tasks=[task], llm_client=llm)
        await node.run(ctx)

        self.assertIn("OUTPUT_EINS", llm.calls[0])
        self.assertIn("OUTPUT_ZWEI", llm.calls[0])


# ───────────────────────────────────────────────────────────────────
# Retry logic
# ───────────────────────────────────────────────────────────────────


class TestRetryLogic(unittest.TestCase):

    @async_test
    async def test_retry_succeeds_after_failures(self):
        """The first attempts fail, the third succeeds."""
        task = SubTask(
            id="T1", phase="p", description="x",
            prompt_template="MARKER", max_retries=3,
        )
        llm = _MockLLMClient(
            responses={"MARKER": "ERFOLG"},
            fail_count={"MARKER": 2},  # the first 2 attempts fail
        )

        ctx = _MinimalCtx()
        node = AnalysisLayerNode(layer_idx=0, tasks=[task], llm_client=llm)
        await node.run(ctx)

        result = ctx.task_results["T1"]
        self.assertEqual(result.state, TaskState.DONE)
        self.assertEqual(result.output, "ERFOLG")
        self.assertEqual(result.retries_used, 2)
        # the LLM was called 3 times
        self.assertEqual(len(llm.calls), 3)

    @async_test
    async def test_failure_after_all_retries(self):
        """If there is no success even after max_retries: FAILED."""
        task = SubTask(
            id="T1", phase="p", description="x",
            prompt_template="MARKER", max_retries=1,
        )
        llm = _MockLLMClient(
            responses={"MARKER": "WIRD-NIE-ERREICHT"},
            fail_count={"MARKER": 99},  # always fail
        )

        ctx = _MinimalCtx()
        node = AnalysisLayerNode(layer_idx=0, tasks=[task], llm_client=llm)
        await node.run(ctx)

        result = ctx.task_results["T1"]
        self.assertEqual(result.state, TaskState.FAILED)
        self.assertIn("mock failure", result.error)
        # max_retries=1 → 2 attempts
        self.assertEqual(len(llm.calls), 2)


# ───────────────────────────────────────────────────────────────────
# Decomposer
# ───────────────────────────────────────────────────────────────────


class TestDecomposer(unittest.TestCase):
    """Tests for the decomposer."""

    @async_test
    async def test_unknown_use_case_returns_empty_plan(self):
        """Unknown use case → empty plan + warning."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="?", use_case="non_existent", preflight_data={},
        )
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(plan.tasks, [])

    @async_test
    async def test_explainer_with_llm_concept_extraction(self):
        """Explainer makes an LLM call and builds the plan from the concept list."""
        # the LLM returns JSON with three concepts
        llm = _MockLLMClient({
            "Identify 3-5 core concepts": (
                '{"concepts": ["Welle-Teilchen-Dualität", '
                '"Heisenberg-Unschärfe", "Quantenverschränkung"]}'
            ),
        })
        decomposer = Decomposer(llm_client=llm)
        plan = await decomposer.decompose(
            query="Quantenmechanik",
            use_case="explainer",
            preflight_data={
                "topic": "Quantenmechanik",
                "audience": "Studierende",
                "length": "kurz (1-2 Seiten)",
            },
        )

        # expected structure: 3 concepts → 3 explanation + 3 example + 1 synthesis
        self.assertEqual(plan.use_case, "explainer")
        self.assertEqual(len(plan.tasks), 7)

        phases = [t.phase for t in plan.tasks]
        self.assertEqual(phases.count("explanation"), 3)
        self.assertEqual(phases.count("example"), 3)
        self.assertEqual(phases.count("synthesis"), 1)

        # topology: examples depend on explanations, synthesis on everything
        synth = [t for t in plan.tasks if t.phase == "synthesis"][0]
        self.assertEqual(len(synth.depends_on), 6)

    @async_test
    async def test_explainer_falls_back_on_invalid_concepts(self):
        """If the LLM delivers no JSON concepts: fall back to the topic."""
        llm = _MockLLMClient({
            "Identify 3-5 core concepts": "kein JSON",
        })
        decomposer = Decomposer(llm_client=llm)
        plan = await decomposer.decompose(
            query="Test",
            use_case="explainer",
            preflight_data={"topic": "Test", "audience": "x", "length": "y"},
        )
        # at least one concept (fallback to the topic)
        n_explanations = sum(1 for t in plan.tasks if t.phase == "explanation")
        self.assertGreaterEqual(n_explanations, 1)

    @async_test
    async def test_peer_review_full_review(self):
        """Peer review with the focus 'full' generates four aspects."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="Review",
            use_case="peer_review",
            preflight_data={
                "manuscript_summary": "Ein Manuskript...",
                "discipline": "Soziologie",
                "review_focus": "full",
            },
        )
        self.assertEqual(plan.use_case, "peer_review")
        # 4 aspects + 1 recommendation
        self.assertEqual(len(plan.tasks), 5)
        # the recommendation depends on all aspects
        rec = [t for t in plan.tasks if t.id == "REC"][0]
        self.assertEqual(len(rec.depends_on), 4)

    @async_test
    async def test_peer_review_focused_methodology(self):
        """Peer review with the focus 'methodology' generates only one aspect."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="peer_review",
            preflight_data={
                "manuscript_summary": "x",
                "discipline": "x",
                "review_focus": "methodology",
            },
        )
        # 1 aspect + 1 recommendation
        self.assertEqual(len(plan.tasks), 2)

    @async_test
    async def test_decision_analysis_builds_matrix(self):
        """Decision analysis builds an options × criteria matrix."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="decision_analysis",
            preflight_data={
                "decision": "Hardware-Kauf",
                "options": "Option A\nOption B\nOption C",
                "criteria": "Kosten\nLeistung",
            },
        )
        # 3 options × 2 criteria = 6 cells + 1 recommendation
        self.assertEqual(len(plan.tasks), 7)
        cells = [t for t in plan.tasks if t.phase == "cell_evaluation"]
        self.assertEqual(len(cells), 6)

    @async_test
    async def test_decision_analysis_uses_default_criteria(self):
        """Without criteria: default list."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="decision_analysis",
            preflight_data={
                "decision": "Test",
                "options": "A\nB",
                # criteria empty
            },
        )
        # 2 options × 4 default criteria = 8 + 1
        self.assertEqual(len(plan.tasks), 9)

    @async_test
    async def test_grant_proposal_six_sections_plus_coherence(self):
        """Grant proposal: 6 sections + coherence check."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="grant_proposal",
            preflight_data={
                "research_question": "Test-Frage",
                "funder": "DFG",
                "duration": "36 Monate",
            },
        )
        sections = [t for t in plan.tasks if t.phase == "section_drafting"]
        self.assertEqual(len(sections), 6)
        # the coherence check depends on all sections
        coh = [t for t in plan.tasks if t.id == "COHERENCE"][0]
        self.assertEqual(len(coh.depends_on), 6)

    @async_test
    async def test_research_design_phase_chain(self):
        """Research design: gap → hypotheses → (method, limitations) → integration."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="research_design",
            preflight_data={
                "research_question": "Test?",
                "discipline": "Soziologie",
                "design_preference": "qualitativ",
            },
        )
        ids = {t.id for t in plan.tasks}
        self.assertEqual(ids, {"GAP", "HYPO", "METHOD", "LIMIT", "DESIGN"})
        # check the topology
        method = next(t for t in plan.tasks if t.id == "METHOD")
        self.assertEqual(method.depends_on, ["HYPO"])
        design = next(t for t in plan.tasks if t.id == "DESIGN")
        self.assertEqual(set(design.depends_on),
                         {"GAP", "HYPO", "METHOD", "LIMIT"})
        # the topological order can be computed
        layers = plan.topological_order()
        self.assertGreater(len(layers), 1)

    @async_test
    async def test_literature_review_with_custom_questions(self):
        """Literature review uses the user's key_questions."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="literature_review",
            preflight_data={
                "topic": "BAföG-Reform",
                "discipline": "Bildungswissenschaft",
                "key_questions": "Frage 1\nFrage 2",
            },
        )
        questions = [t for t in plan.tasks if t.phase == "question_synthesis"]
        self.assertEqual(len(questions), 2)

        # The plan includes obtaining literature and a meta-synthesis, so that
        # the review does not come from the model's knowledge alone.
        ids = [t.id for t in plan.tasks]
        self.assertEqual(ids, ["QUERIES", "SEARCH", "Q_1", "Q_2", "META"])

        search = next(t for t in plan.tasks if t.id == "SEARCH")
        self.assertEqual(search.executor, "literature_search")
        self.assertTrue(search.search_config["expand_citations"])
        # every question synthesis must see the literature found
        for q in questions:
            self.assertEqual(q.param_deps.get("hits"), "SEARCH")

    @async_test
    async def test_literature_review_default_questions(self):
        """Without key_questions: three default questions."""
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="literature_review",
            preflight_data={"topic": "x", "discipline": "x"},
        )
        questions = [t for t in plan.tasks if t.phase == "question_synthesis"]
        self.assertEqual(len(questions), 3)

    @async_test
    async def test_literature_finder_searches_and_assesses(self):
        """The mode does not stop at the search queries.

        SEARCH runs the queries against the literature APIs, ASSESS
        evaluates the hits.
        """
        decomposer = Decomposer(llm_client=_MockLLMClient({}))
        plan = await decomposer.decompose(
            query="x",
            use_case="literature_finder",
            preflight_data={"topic": "Test"},
        )
        ids = [t.id for t in plan.tasks]
        self.assertEqual(
            ids, ["STRATEGY", "QUERIES", "SEARCH", "ASSESS", "SELECTION"])

        search = next(t for t in plan.tasks if t.id == "SEARCH")
        self.assertEqual(search.executor, "literature_search")
        self.assertTrue(search.show_in_report)

        # SELECTION renders the selection with the bibliographic details from
        # the API — without an LLM call, so that authors, journal and DOI are
        # not copied by the model.
        sel = next(t for t in plan.tasks if t.id == "SELECTION")
        self.assertEqual(sel.executor, "render_selection")

        # estimated_calls only counts LLM tasks — search and rendering cost
        # no model request.
        self.assertEqual(plan.estimated_calls, 3)

    @async_test
    async def test_decomposer_swallows_internal_errors(self):
        """If an LLM call crashes, the decomposer returns an empty plan instead
        of crashing the pipeline."""
        class _Broken:
            async def primary_complete_json(self, msgs, **kw):
                raise RuntimeError("LLM down")
            async def harvest_complete(self, msgs, **kw):
                raise RuntimeError("LLM down")

        decomposer = Decomposer(llm_client=_Broken())
        plan = await decomposer.decompose(
            query="x",
            use_case="explainer",
            preflight_data={"topic": "x", "audience": "y", "length": "z"},
        )
        # empty plan, no crash
        self.assertEqual(plan.tasks, [])

    @async_test
    async def test_parse_lines_helper(self):
        """_parse_lines cleans up bullets, numbers and empty lines."""
        out = Decomposer._parse_lines(
            "- Erste Option\n"
            "* Zweite Option\n"
            "1. Dritte Option\n"
            "\n"
            "  4) Vierte Option  \n"
            "• Fünfte Option\n"
        )
        self.assertEqual(out, [
            "Erste Option", "Zweite Option", "Dritte Option",
            "Vierte Option", "Fünfte Option",
        ])

    @async_test
    async def test_all_use_cases_produce_resolvable_prompts(self):
        """Regression test: all analysis use cases build plans whose prompt
        templates can be resolved completely.

        Prevents bugs of the form: prompt_template contains {var}, but the
        variable is not defined in prompt_params/param_deps/
        concat_param_deps. Such bugs would show up at run time as unresolved
        {var} markers in the LLM prompt — hard to debug.
        """
        import re
        from src.pipeline.analysis_pipeline import (
            ANALYSIS_USE_CASES, AnalysisLayerNode,
        )

        # typical mandatory inputs per use case (all mandatory fields filled)
        inputs = {
            "explainer": {
                "topic": "Test-Thema", "audience": "Studierende",
                "length": "kurz",
            },
            "peer_review": {
                "manuscript_summary": "Manuskript",
                "discipline": "Soziologie",
                "review_focus": "full",
            },
            "decision_analysis": {
                "decision": "Test", "options": "A\nB", "criteria": "K1",
            },
            "grant_proposal": {
                "research_question": "?", "funder": "DFG",
                "duration": "36 Monate",
            },
            "research_design": {
                "research_question": "?", "discipline": "x",
            },
            "literature_review": {
                "topic": "x", "discipline": "y",
            },
            "literature_finder": {"topic": "x"},
        }

        # mock LLM for use cases with a preliminary LLM call (explainer)
        llm = _MockLLMClient({
            "Identify 3-5 core concepts": (
                '{"concepts": ["A", "B"]}'
            ),
        })
        decomposer = Decomposer(llm)

        for uc in sorted(ANALYSIS_USE_CASES):
            plan = await decomposer.decompose("q", uc, inputs[uc])
            self.assertGreater(
                len(plan.tasks), 0,
                f"{uc}: empty plan — decomposer broken?",
            )

            # pre-fill all dependency tasks with state DONE
            results = {
                dep: TaskResult(
                    task_id=dep, state=TaskState.DONE,
                    output=f"[{dep}]",
                )
                for t in plan.tasks for dep in t.depends_on
            }

            for t in plan.tasks:
                try:
                    resolved = AnalysisLayerNode._resolve_prompt(t, results)
                except Exception as e:
                    self.fail(
                        f"{uc}/{t.id}: _resolve_prompt crash: {e}",
                    )
                # no unresolved {var} markers
                unresolved = re.findall(r'\{[a-z_]+\}', resolved)
                self.assertEqual(
                    unresolved, [],
                    f"{uc}/{t.id}: unresolved vars {unresolved} in the "
                    f"resolved prompt:\n{resolved[:200]}",
                )
                # no unresolved {dep:...} markers
                unresolved_deps = re.findall(r'\{dep:[^}]+\}', resolved)
                self.assertEqual(
                    unresolved_deps, [],
                    f"{uc}/{t.id}: unresolved deps {unresolved_deps}",
                )


# ───────────────────────────────────────────────────────────────────
# AnalysisPipelineRunner
# ───────────────────────────────────────────────────────────────────


class TestAnalysisPipelineRunner(unittest.TestCase):

    async def _noop_cb(self, *args, **kwargs):
        pass

    @async_test
    async def test_decompose_only_sets_plan_and_status(self):
        runner = AnalysisPipelineRunner(
            llm_client=_MockLLMClient({
                "Identify 3-5 core concepts": (
                    '{"concepts": ["A", "B"]}'
                ),
            }),
            progress_callback=self._noop_cb,
        )
        ctx = _MinimalCtx()
        ctx.use_case = "explainer"
        ctx.preflight_data = {
            "topic": "T", "audience": "A", "length": "L",
        }
        await runner.decompose_only(ctx)

        # with the real decomposer: the plan is filled
        self.assertIsNotNone(ctx.task_plan)
        self.assertGreater(len(ctx.task_plan.tasks), 0)
        self.assertEqual(ctx.status, "plan_ready")

    @async_test
    async def test_run_with_unknown_use_case_errors_clearly(self):
        """Unknown use case → empty plan → clear error message."""
        runner = AnalysisPipelineRunner(
            llm_client=_MockLLMClient({}),
            progress_callback=self._noop_cb,
        )
        ctx = _MinimalCtx()
        ctx.use_case = "non_existent_use_case"
        await runner.run(ctx, skip_decompose=False)

        self.assertEqual(ctx.status, "error")
        self.assertIn("decomposer", ctx.error_message)
        self.assertIn("not implemented", ctx.error_message)

    @async_test
    async def test_run_with_prefab_plan_executes_layers(self):
        """With a hard-wired TaskPlan the DAG execution runs."""
        # 3 tasks: T1 → T2, T1 → T3 (T2 and T3 depend on T1)
        plan = TaskPlan(
            use_case="explainer",
            tasks=[
                SubTask(id="T1", phase="p", description="x",
                        prompt_template="MARKER_1"),
                SubTask(id="T2", phase="p", description="x",
                        prompt_template="MARKER_2 — {dep:T1}",
                        depends_on=["T1"]),
                SubTask(id="T3", phase="p", description="x",
                        prompt_template="MARKER_3 — {dep:T1}",
                        depends_on=["T1"]),
            ],
        )
        ctx = _MinimalCtx()
        ctx.task_plan = plan

        llm = _MockLLMClient(responses={
            "MARKER_1": "out-T1",
            "MARKER_2": "out-T2",
            "MARKER_3": "out-T3",
        })
        runner = AnalysisPipelineRunner(
            llm_client=llm, progress_callback=self._noop_cb,
        )
        await runner.run(ctx, skip_decompose=True)

        self.assertEqual(ctx.status, "done")
        # all 3 tasks: DONE
        self.assertEqual(ctx.task_results["T1"].state, TaskState.DONE)
        self.assertEqual(ctx.task_results["T2"].state, TaskState.DONE)
        self.assertEqual(ctx.task_results["T3"].state, TaskState.DONE)
        # T2 and T3 got the T1 output substituted
        self.assertIn("out-T1", llm.calls[1])  # layer 2, call 1 (T2 or T3)
        self.assertIn("out-T1", llm.calls[2])  # layer 2, call 2

    @async_test
    async def test_stop_check_aborts_pipeline(self):
        """An external stop check prevents execution."""
        plan = TaskPlan(
            use_case="explainer",
            tasks=[SubTask(id="T1", phase="p", description="x",
                           prompt_template="MARKER")],
        )
        ctx = _MinimalCtx()
        ctx.task_plan = plan
        llm = _MockLLMClient({"MARKER": "ok"})

        runner = AnalysisPipelineRunner(
            llm_client=llm, progress_callback=self._noop_cb,
            stop_check=lambda: True,  # always True
        )
        await runner.run(ctx, skip_decompose=True)

        # The pipeline did not run — status either cancelled or
        # T1 SKIPPED. Acceptable: what matters is that the LLM was not called
        # (the exact semantics depend on the engine; mainly: the stop signal
        #  was respected)
        # at least: not "done"
        self.assertNotEqual(ctx.status, "done",
                            f"status should not be 'done' with stop_check, "
                            f"was: {ctx.status}")

    @async_test
    async def test_stop_check_aborts_mid_pipeline(self):
        """A stop signal in the middle of the pipeline (after the first task)
        must take effect before all further tasks have run.

        stop_check must be called periodically, not only once before the
        start — otherwise the user could not stop long pipelines.
        """
        # 5 tasks in a chain, every LLM call waits 0.05 s
        plan = TaskPlan(
            use_case="explainer",
            tasks=[
                SubTask(id=f"T{i}", phase="p", description=f"T{i}",
                        prompt_template=f"M{i} {{dep:T{i-1}}}"
                                         if i > 0 else f"M{i}",
                        depends_on=[f"T{i-1}"] if i > 0 else [])
                for i in range(5)
            ],
        )
        ctx = _MinimalCtx()
        ctx.task_plan = plan

        import time as _time
        call_counter = [0]

        class _SlowLLM:
            async def harvest_complete(self, msgs, **kw):
                call_counter[0] += 1
                await asyncio.sleep(0.05)
                return f"out-{call_counter[0]}"
            primary_complete = harvest_complete

        # stop after 130 ms — should stop after ~3 tasks
        start = _time.monotonic()
        runner = AnalysisPipelineRunner(
            llm_client=_SlowLLM(), progress_callback=self._noop_cb,
            stop_check=lambda: _time.monotonic() - start > 0.13,
        )
        await runner.run(ctx, skip_decompose=True)

        # stopped: neither all 5 tasks ran, nor none — and the tasks that
        # ran succeeded (a failing double would also stop the chain early)
        done = [t for t, r in ctx.task_results.items() if r.state == TaskState.DONE]
        self.assertTrue(done, "no task completed — the test double is broken")
        self.assertLess(call_counter[0], 5,
                        "all 5 tasks ran — the stop signal "
                        "did NOT take effect")
        self.assertGreater(call_counter[0], 0,
                           "not a single task ran — "
                           "the pipeline stopped too early")

    @async_test
    async def test_final_report_synthesis_from_final_task(self):
        """With status=done, ctx.final_report must be filled from the final
        task's output (otherwise ctx.final_report would stay empty).
        """
        # plan: T1 → T2, with T2 as the final task
        plan = TaskPlan(
            use_case="explainer",
            tasks=[
                SubTask(id="T1", phase="p1", description="Vorarbeit",
                        prompt_template="MARKER_1"),
                SubTask(id="T2", phase="p2", description="Synthese",
                        prompt_template="MARKER_2 {dep:T1}",
                        depends_on=["T1"]),
            ],
        )
        ctx = _MinimalCtx()
        ctx.task_plan = plan
        ctx.preflight_data = {"topic": "Mein Thema"}

        llm = _MockLLMClient({
            "MARKER_1": "Vorarbeit-Output",
            "MARKER_2": "FINALE-SYNTHESE",
        })
        runner = AnalysisPipelineRunner(
            llm_client=llm, progress_callback=self._noop_cb,
        )
        await runner.run(ctx, skip_decompose=True)

        self.assertEqual(ctx.status, "done")
        self.assertIsNotNone(ctx.final_report)
        # the final task output must be included
        self.assertIn("FINALE-SYNTHESE", ctx.final_report)
        # title from preflight_data
        self.assertIn("Mein Thema", ctx.final_report)

    @async_test
    async def test_final_report_warns_on_failed_tasks(self):
        """If tasks fail, the report must make that transparent instead of
        silently finishing with incomplete content.
        """
        plan = TaskPlan(
            use_case="explainer",
            tasks=[
                SubTask(id="T1", phase="p", description="Wichtige Vorarbeit",
                        prompt_template="MARKER_FAIL", max_retries=0),
                SubTask(id="T2", phase="p", description="Final",
                        prompt_template="MARKER_OK"),
            ],
        )
        ctx = _MinimalCtx()
        ctx.task_plan = plan
        ctx.preflight_data = {"topic": "Test"}

        # T1 raises, T2 runs normally
        class _SelectiveLLM:
            calls = []
            async def harvest_complete(self, msgs, **kw):
                content = msgs[0]["content"]
                _SelectiveLLM.calls.append(content)
                if "MARKER_FAIL" in content:
                    raise RuntimeError("simulated error")
                return "T2-Output"
            primary_complete = harvest_complete

        runner = AnalysisPipelineRunner(
            llm_client=_SelectiveLLM(), progress_callback=self._noop_cb,
        )
        await runner.run(ctx, skip_decompose=True)

        self.assertEqual(ctx.status, "done")
        # the report should contain the error notice
        self.assertIn("Note", ctx.final_report)
        self.assertIn("Wichtige Vorarbeit", ctx.final_report)


if __name__ == "__main__":
    unittest.main()
