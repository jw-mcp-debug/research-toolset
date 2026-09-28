"""
Test package. Loads `conftest` first of all, so that sandbox stubs (httpx
etc.) are set up before any test imports a module from `src.*`.

pytest would pick up conftest.py automatically; this import makes the stubs
available when a single test module is imported directly as well. The
supported test runner is pytest (several tests use pytest fixtures and
plain async test functions).
"""

from tests import conftest  # noqa: F401
