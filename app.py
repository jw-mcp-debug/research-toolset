#!/usr/bin/env python3
"""
Research tool — entry point
===========================
Autonomous web research with LLM-assisted analysis and synthesis.

Privacy-related start-up settings:
  - Telemetry switches are set BEFORE `import gradio` (process level).
  - The theme uses system fonts only; no Google Fonts.
  - `share=False`, `pwa=False`, `flagging_mode="never"`.
  - Upload limits, path allow/block lists; Gradio cleans up uploads and
    downloads, stored runs are deleted after CLEANUP_MAX_AGE_DAYS.
  - HTTPS via an SSL certificate or a reverse proxy.
"""

# ════════════════════════════════════════════════════════════════════
# 1. Telemetry switches and paths — set BEFORE `import gradio`
# ════════════════════════════════════════════════════════════════════
import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("GRADIO_FLAGGING_MODE", "never")
# GRADIO_TEMP_DIR: dedicated directory instead of /tmp.
# Overridden by AppConfig.from_env() when set there.
# The default lives in the system temp directory so that a start without
# root privileges (IDE, local venv) works. The Docker image sets
# APP_TEMP_DIR=/var/cache/research-toolset.
import tempfile


def _env_renamed(new: str, old: str, default=None):
    """Read `new`; fall back to the former name `old` for existing installations."""
    value = os.environ.get(new)
    if value is None and os.environ.get(old) is not None:
        print(f"Note: {old} is deprecated, please rename it to {new}.", file=sys.stderr)
        value = os.environ.get(old)
    return default if value is None else value


os.environ.setdefault(
    "GRADIO_TEMP_DIR",
    _env_renamed(
        "APP_TEMP_DIR", "RECHERCHE_TEMP_DIR",
        os.path.join(tempfile.gettempdir(), "research-toolset", "cache"),
    ),
)

# ════════════════════════════════════════════════════════════════════
# 2. Imports
# ════════════════════════════════════════════════════════════════════
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Imported explicitly here, after the telemetry switches above, so that the
# ordering is visible (and checked by tests/test_privacy_config.py).
import gradio as gr  # noqa: F401,E402
from src.about import TOOL_NAME, VERSION
from src.config import AppConfig
from src.ui.gradio_app import create_app, APP_THEME
from src.ui.css import CUSTOM_CSS


# ════════════════════════════════════════════════════════════════════
# 3. Main
# ════════════════════════════════════════════════════════════════════
def main():
    config = AppConfig.from_env()

    logger.info("=" * 55)
    logger.info(f"{TOOL_NAME} {VERSION}")
    logger.info("=" * 55)
    logger.info(f"Server:       {config.ui.server_name}:{config.ui.server_port}")
    logger.info(f"Path:         {config.ui.root_path}")
    logger.info(f"Primary LLM:  {config.llm.primary.model_name} @ {config.llm.primary.api_base}")
    logger.info(f"Harvest LLM:  {config.llm.harvest.model_name} @ {config.llm.harvest.api_base}")
    logger.info(f"Harvest ‖:    {config.llm.harvest_max_parallel} parallel")
    logger.info(f"SearXNG:      {config.connectors.searxng.base_url}")
    logger.info(f"Data:         {config.pipeline.data_dir}")
    logger.info(f"Temp:         {os.environ.get('GRADIO_TEMP_DIR')}")
    logger.info("=" * 55)

    # Create the directory for stored research data.
    # Gradio cleans up user uploads and download caches automatically
    # via `delete_cache=(3600, 14400)` in gr.Blocks(...)
    # and its component lifecycle management.
    os.makedirs(config.pipeline.data_dir, exist_ok=True)

    # Fail at start-up, not mid-report, if OUTPUT_LANGUAGES names a
    # language without a (valid) catalog.
    from src.output_language import enabled_languages
    logger.info("Output languages: %s", ", ".join(enabled_languages()))

    # Stored runs older than CLEANUP_MAX_AGE_DAYS are deleted at start-up
    # and periodically afterwards (0 disables this).
    from src.core.retention import start_retention_worker
    start_retention_worker(
        config.pipeline.data_dir, config.pipeline.cleanup_max_age_days,
    )

    # ──────────────────────────────────────────────────────────────
    # App + queue (configure before launch!)
    # ──────────────────────────────────────────────────────────────
    app = create_app(config)
    app.queue(
        default_concurrency_limit=4,
        max_size=64,
        api_open=False,            # do not bypass the queue via REST
        status_update_rate="auto",
    )

    # ──────────────────────────────────────────────────────────────
    # SSL configuration (optional, otherwise behind a reverse proxy)
    # ──────────────────────────────────────────────────────────────
    ssl_kwargs = {}
    ssl_certfile = _env_renamed("SERVER_SSL_CERTFILE", "RECHERCHE_SSL_CERTFILE")
    ssl_keyfile = _env_renamed("SERVER_SSL_KEYFILE", "RECHERCHE_SSL_KEYFILE")
    if ssl_certfile and ssl_keyfile:
        ssl_kwargs.update({
            "ssl_certfile": ssl_certfile,
            "ssl_keyfile": ssl_keyfile,
            "ssl_keyfile_password": _env_renamed("SERVER_SSL_PASSWORD", "RECHERCHE_SSL_PASSWORD"),
            "ssl_verify": True,
        })
        logger.info("SSL active: %s", ssl_certfile)
    else:
        logger.info("No SSL — serve HTTPS via a reverse proxy")

    # ──────────────────────────────────────────────────────────────
    # Auth (optional via Env)
    # ──────────────────────────────────────────────────────────────
    auth_kwargs = {}
    auth_user = _env_renamed("SERVER_AUTH_USER", "RECHERCHE_AUTH_USER")
    auth_pass = _env_renamed("SERVER_AUTH_PASSWORD", "RECHERCHE_AUTH_PASSWORD")
    if auth_user and auth_pass:
        auth_kwargs["auth"] = (auth_user, auth_pass)
        logger.info("Auth active for user %r", auth_user)

    # ──────────────────────────────────────────────────────────────
    # Launch — privacy-preserving parameters
    # ──────────────────────────────────────────────────────────────
    app.launch(
        # Theme and CSS — in Gradio 6 passed to launch(),
        # no longer to gr.Blocks(). System fonts instead of Google Fonts.
        theme=APP_THEME,
        css=CUSTOM_CSS,

        # Server
        server_name=config.ui.server_name,
        server_port=config.ui.server_port,
        root_path=config.ui.root_path,
        **ssl_kwargs,

        # Auth
        **auth_kwargs,

        # File-Limits
        max_file_size="20mb",
        # allowed_paths empty = only the defaults (cwd, tempfile, cache).
        # If static assets are needed, add them here explicitly.
        allowed_paths=[],
        # Safety net against path traversal. /home is deliberately not
        # listed: Gradio checks the block list before its own created
        # files, so a temp directory under /home would otherwise block
        # every download and upload.
        blocked_paths=["/etc", "/root"],

        # Privacy / Logging
        show_error=False,         # no tracebacks in the browser modal
        quiet=True,               # less server output
        footer_links=[],          # no "Built with Gradio", no API docs

        # prohibitions
        share=False,              # no external tunnels
        pwa=False,                # PWA off
        ssr_mode=False,           # SSR off (needs Node 20+)
        inbrowser=False,
        strict_cors=True,         # default, do not switch off
    )


if __name__ == "__main__":
    main()
