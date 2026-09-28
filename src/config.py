"""
Central configuration of the research tool.
All values are read from environment variables, with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


# ─── LLM ────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    model_name: str = "llm"
    max_context_tokens: int = 260_000
    max_output_tokens: int = 32_768
    default_temperature: float = 0.3
    default_top_p: float = 0.6
    timeout: float = 120.0
    stream: bool = True

    def is_kimi(self) -> bool:
        return "kimi" in self.model_name.lower()


@dataclass
class DualLLMConfig:
    """Configuration of the primary and harvest LLM."""
    primary: LLMConfig = field(default_factory=LLMConfig)
    harvest: LLMConfig = field(default_factory=LLMConfig)
    harvest_max_parallel: int = 3

    @classmethod
    def from_env(cls) -> "DualLLMConfig":
        primary = LLMConfig(
            api_base=_env("LLM_API_BASE", "http://localhost:8000/v1"),
            api_key=_env("LLM_API_KEY", "not-needed"),
            model_name=_env("LLM_MODEL_NAME", "llm"),
            max_context_tokens=_env_int("LLM_MAX_CONTEXT_TOKENS", 260_000),
            max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 32_768),
            default_temperature=_env_float("LLM_DEFAULT_TEMPERATURE", 0.3),
            default_top_p=_env_float("LLM_DEFAULT_TOP_P", 0.6),
            timeout=_env_float("LLM_TIMEOUT", 120.0),
        )

        # Harvest LLM: falls back to the primary
        harvest_base = _env("HARVEST_LLM_API_BASE", primary.api_base)
        harvest_model = _env("HARVEST_LLM_MODEL_NAME", primary.model_name)
        harvest_key = _env("HARVEST_LLM_API_KEY", primary.api_key)

        harvest = LLMConfig(
            api_base=harvest_base,
            api_key=harvest_key,
            model_name=harvest_model,
            max_context_tokens=_env_int("HARVEST_LLM_MAX_CONTEXT_TOKENS", 32_000),
            max_output_tokens=_env_int("HARVEST_LLM_MAX_OUTPUT_TOKENS", 4_096),
            default_temperature=_env_float("HARVEST_LLM_DEFAULT_TEMPERATURE", 0.2),
            timeout=_env_float("HARVEST_LLM_TIMEOUT", 90.0),
        )

        # Parallelism: same model → less parallel
        #
        # With a separate (and faster) harvest model, the semaphore is the
        # only global control for that model: fact extraction as well as all
        # classifier calls go through `harvest_complete`. With 31 sources,
        # 10 parallel calls meant 4 waves; 16 turn that into 2.
        #
        #
        # The value is deliberately moderate and can be overridden via env:
        # if the endpoint is overloaded it answers with 503, and the back-off
        # (5/10/20 s) makes the run slower instead of faster. When raising it,
        # watch the retry warnings in the log.
        default_parallel = 3 if harvest_model == primary.model_name else 16
        harvest_max_parallel = _env_int("HARVEST_MAX_PARALLEL", default_parallel)

        return cls(
            primary=primary,
            harvest=harvest,
            harvest_max_parallel=harvest_max_parallel,
        )


# ─── Connectors ────────────────────────────────────────────────────

@dataclass
class SearXNGConfig:
    base_url: str = "http://searxng:8080"
    timeout: float = 30.0
    max_results: int = 10


@dataclass
class GitHubConfig:
    token: str = ""
    max_requests_per_hour: int = 5000
    enabled: bool = True


@dataclass
class GitLabConfig:
    base_url: str = ""
    token: str = ""
    enabled: bool = False


@dataclass
class WebScraperConfig:
    timeout: float = 30.0
    max_content_length: int = 500_000  # characters
    use_playwright_fallback: bool = False
    playwright_min_content_length: int = 100
    # Hosts that may be fetched even though they resolve to internal
    # addresses (e.g. an intranet wiki). Comma-separated in the env.
    allowed_internal_hosts: str = ""


@dataclass
class ElasticsearchConfig:
    """Configuration of the Elasticsearch connector."""
    base_url: str = ""                # e.g. "https://elastic.example.com:9200"
    index: str = ""                   # e.g. "website-content"
    api_key: str = ""                 # API key for authentication
    username: str = ""                # Basic auth (alternative to an API key)
    password: str = ""
    # Field mapping: what are the fields called in the index?
    field_title: str = "title"
    field_body: str = "body"          # main content field (text)
    field_url: str = "url"            # URL of the page
    field_path: str = ""              # optional path (e.g. for CMS pages)
    # Optional filters
    site_base_url: str = ""           # e.g. "https://www.example.com" — for URL routing
    max_results: int = 10
    enabled: bool = False


@dataclass
class PersonDirectoryConfig:
    """Local person directory (SQLite, see docs for the schema).

    Embedding and reranking use the shared EMBEDDER_* / RERANKER_*
    settings of the pipeline.
    """
    db_path: str = ""
    enabled: bool = False


@dataclass
class ConnectorsConfig:
    searxng: SearXNGConfig = field(default_factory=SearXNGConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    gitlab: GitLabConfig = field(default_factory=GitLabConfig)
    web_scraper: WebScraperConfig = field(default_factory=WebScraperConfig)
    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    directory: PersonDirectoryConfig = field(default_factory=PersonDirectoryConfig)

    @classmethod
    def from_env(cls) -> "ConnectorsConfig":
        return cls(
            searxng=SearXNGConfig(
                base_url=_env("SEARXNG_BASE_URL", "http://searxng:8080"),
                timeout=_env_float("SEARXNG_TIMEOUT", 30.0),
                max_results=_env_int("SEARXNG_MAX_RESULTS", 10),
            ),
            github=GitHubConfig(
                token=_env("GITHUB_TOKEN", ""),
                enabled=_env_bool("GITHUB_ENABLED", True),
            ),
            gitlab=GitLabConfig(
                base_url=_env("GITLAB_BASE_URL", ""),
                token=_env("GITLAB_TOKEN", ""),
                enabled=bool(_env("GITLAB_BASE_URL", "")),
            ),
            web_scraper=WebScraperConfig(
                timeout=_env_float("FETCH_TIMEOUT", 30.0),
                use_playwright_fallback=_env_bool("USE_PLAYWRIGHT", False),
                allowed_internal_hosts=_env("FETCH_ALLOWED_INTERNAL_HOSTS", ""),
            ),
            elasticsearch=ElasticsearchConfig(
                base_url=_env("ELASTIC_BASE_URL", ""),
                index=_env("ELASTIC_INDEX", ""),
                api_key=_env("ELASTIC_API_KEY", ""),
                username=_env("ELASTIC_USERNAME", ""),
                password=_env("ELASTIC_PASSWORD", ""),
                field_title=_env("ELASTIC_FIELD_TITLE", "title"),
                field_body=_env("ELASTIC_FIELD_BODY", "body"),
                field_url=_env("ELASTIC_FIELD_URL", "url"),
                field_path=_env("ELASTIC_FIELD_PATH", ""),
                site_base_url=_env("ELASTIC_SITE_BASE_URL", ""),
                max_results=_env_int("ELASTIC_MAX_RESULTS", 10),
                enabled=bool(_env("ELASTIC_BASE_URL", "")),
            ),
            directory=PersonDirectoryConfig(
                db_path=_env("PERSON_DIRECTORY_DB", ""),
                enabled=bool(_env("PERSON_DIRECTORY_DB", ""))
                        and Path(_env("PERSON_DIRECTORY_DB", "")).exists(),
            ),
        )


# ─── Pipeline ───────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    max_rounds: int = 5
    max_sources_per_round: int = 20
    max_parallel_fetches: int = 10
    data_dir: str = "data/runs"
    cleanup_max_age_days: int = 30
    # ── Reranker (Cross-Encoder, /rerank-API) ──
    reranker_base_url: str = ""
    reranker_model: str = ""
    reranker_api_key: str = ""
    # ── Embedder (OpenAI-compatible, /embeddings) ──
    embedder_base_url: str = ""
    embedder_model: str = ""
    embedder_api_key: str = ""
    # Threshold for embedding-based de-duplication of extracts.
    # cosine ≥ threshold ⇒ treated as duplicate. 0.93 = conservative.
    extract_dedup_threshold: float = 0.93
    # ── Search fan-out ──
    # Upper bound of web searches per round. Since searches are cheap (meta search)
    # and the reranker picks the top `max_sources_per_round` for fetching,
    # the candidate pool may be larger than the fetch cap.
    max_web_searches: int = 40
    max_web_searches_institution: int = 60

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            max_rounds=_env_int("MAX_RESEARCH_ROUNDS", 5),
            max_sources_per_round=_env_int("MAX_SOURCES_PER_ROUND", 20),
            max_parallel_fetches=_env_int("MAX_PARALLEL_FETCHES", 10),
            data_dir=_env("DATA_DIR", "data/runs"),
            cleanup_max_age_days=_env_int("CLEANUP_MAX_AGE_DAYS", 30),
            reranker_base_url=_env("RERANKER_BASE_URL", ""),
            reranker_model=_env("RERANKER_MODEL", ""),
            reranker_api_key=_env("RERANKER_API_KEY", ""),
            embedder_base_url=_env("EMBEDDER_BASE_URL", ""),
            embedder_model=_env("EMBEDDER_MODEL", ""),
            embedder_api_key=_env("EMBEDDER_API_KEY", ""),
            extract_dedup_threshold=_env_float(
                "EXTRACT_DEDUP_THRESHOLD", 0.93
            ),
            max_web_searches=_env_int("MAX_WEB_SEARCHES", 40),
            max_web_searches_institution=_env_int("MAX_WEB_SEARCHES_INSTITUTION", 60),
        )


# ─── UI ─────────────────────────────────────────────────────────────

@dataclass
class UIConfig:
    # Reachable only locally unless configured otherwise. The Docker
    # image sets GRADIO_SERVER_NAME=0.0.0.0.
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    root_path: str = ""
    max_documents: int = 25
    min_paste_doc_length: int = 500


# ─── Document processing ──────────────────────────────────────────

@dataclass
class ProcessorConfig:
    remove_headers: bool = True
    remove_footers: bool = True
    remove_page_numbers: bool = True
    deduplicate: bool = True
    similarity_threshold: float = 0.95
    min_paragraph_length: int = 20
    pdf_strategy: str = "fast"


# ─── Overall ─────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    llm: DualLLMConfig = field(default_factory=DualLLMConfig)
    connectors: ConnectorsConfig = field(default_factory=ConnectorsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm=DualLLMConfig.from_env(),
            connectors=ConnectorsConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            ui=UIConfig(
                server_name=_env("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=_env_int("GRADIO_SERVER_PORT", 7860),
                root_path=_env("ROOT_PATH", ""),
                max_documents=_env_int("MAX_DOCUMENTS", 25),
            ),
            processor=ProcessorConfig(
                pdf_strategy=_env("PDF_STRATEGY", "fast"),
            ),
        )


# Document formats
PLAIN_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".py"}
RICH_DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".html", ".htm", ".rtf",
}
DOCUMENT_EXTENSIONS = PLAIN_TEXT_EXTENSIONS | RICH_DOCUMENT_EXTENSIONS
