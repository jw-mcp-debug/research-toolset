"""
Test configuration and sandbox stubs.

Loaded automatically by pytest, the supported test runner, and imported
first by `tests/__init__.py`.

Purpose: in CI environments where third-party dependencies such as `httpx`
are not installed, this stub provides a sufficient mock module, so that
`src.llm.client` and thus the orchestrator can be imported — the tests
themselves do not use httpx functionality (they mock their own LLM
adapter, MockLLM).
"""

import sys
import types


def _ignore_local_env_file() -> None:
    """Tests must not depend on a developer's local `.env`.

    `src.config` calls `load_dotenv()` at import time. A local `.env` (as the
    README asks for) would otherwise change defaults such as the output
    language or the number of research rounds and make tests fail. This runs
    before any test module imports `src`.
    """
    try:
        import dotenv
    except ImportError:
        return
    dotenv.load_dotenv = lambda *args, **kwargs: False


_ignore_local_env_file()


def _ensure_httpx_stub() -> None:
    try:
        import httpx  # noqa: F401  — the real package wins whenever installed
        return
    except ImportError:
        pass

    httpx_stub = types.ModuleType("httpx")

    class _StubClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aclose(self):
            pass
        async def post(self, *args, **kwargs):
            raise RuntimeError("httpx stub: post() is not allowed in tests")
        async def get(self, *args, **kwargs):
            raise RuntimeError("httpx stub: get() is not allowed in tests")

    class _StubResponse:
        status_code = 200
        text = ""
        def json(self):
            return {}
        def raise_for_status(self):
            pass

    httpx_stub.AsyncClient = _StubClient
    httpx_stub.Client = _StubClient
    httpx_stub.Response = _StubResponse
    httpx_stub.Timeout = lambda *a, **k: None
    httpx_stub.HTTPError = type("HTTPError", (Exception,), {})
    httpx_stub.RequestError = type("RequestError", (Exception,), {})
    httpx_stub.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
    httpx_stub.ReadTimeout = type("ReadTimeout", (Exception,), {})
    httpx_stub.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
    httpx_stub.ConnectError = type("ConnectError", (Exception,), {})
    httpx_stub.WriteError = type("WriteError", (Exception,), {})
    httpx_stub.RemoteProtocolError = type("RemoteProtocolError", (Exception,), {})
    httpx_stub.PoolTimeout = type("PoolTimeout", (Exception,), {})

    sys.modules["httpx"] = httpx_stub


_ensure_httpx_stub()

def _ensure_openai_stub() -> None:
    try:
        import openai  # noqa: F401  — the real package wins whenever installed
        return
    except ImportError:
        pass

    openai_stub = types.ModuleType("openai")

    class _StubAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._not_called),
            )
        async def _not_called(self, *args, **kwargs):
            raise RuntimeError("openai stub: not allowed in tests")

    openai_stub.AsyncOpenAI = _StubAsyncOpenAI
    openai_stub.OpenAI = _StubAsyncOpenAI
    openai_stub.APIError = type("APIError", (Exception,), {})
    openai_stub.APIConnectionError = type("APIConnectionError", (Exception,), {})
    openai_stub.APITimeoutError = type("APITimeoutError", (Exception,), {})
    openai_stub.RateLimitError = type("RateLimitError", (Exception,), {})
    openai_stub.AuthenticationError = type("AuthenticationError", (Exception,), {})
    openai_stub.BadRequestError = type("BadRequestError", (Exception,), {})

    sys.modules["openai"] = openai_stub


_ensure_openai_stub()


# Several test modules install MagicMock stand-ins for internal modules
# ("if 'src.config' not in sys.modules: ...") at import time. Whichever
# test module pytest imports first would otherwise decide for the whole
# session whether the real modules or the stand-ins are used, which made
# results depend on collection order. Importing the real modules here,
# before any test module is collected, keeps those guards inactive.
import src.config  # noqa: E402,F401
import src.connectors.rate_limiter  # noqa: E402,F401
import src.connectors.base  # noqa: E402,F401
import src.llm.client  # noqa: E402,F401
