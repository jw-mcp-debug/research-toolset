"""SSRF protection: IP normalisation, DNS resolution, redirect guard."""

import asyncio
import socket
from unittest import mock

import pytest

from src.core.url_security import (
    UnsafeURLError,
    assert_safe_resolved,
    is_safe_public_url,
    make_request_guard,
    parse_allowed_hosts,
)


@pytest.mark.parametrize("url", [
    "http://127.1/",
    "http://0x7f000001/",
    "http://2130706433/",
    "http://0177.0.0.1/",
    "http://[::ffff:127.0.0.1]/",
    "http://100.64.0.1/",
    "http://10.1/",
])
def test_numeric_ip_forms_are_rejected(url):
    ok, _ = is_safe_public_url(url)
    assert not ok


@pytest.mark.parametrize("url", [
    "https://example.org/",
    "https://abc.de/",          # hex-looking hostname stays a hostname
    "http://8.8.8.8/",
])
def test_public_targets_pass_lexical_check(url):
    ok, reason = is_safe_public_url(url)
    assert ok, reason


def _fake_getaddrinfo(addresses):
    async def _gai(host, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0))
                for a in addresses]
    return _gai


def _run(coro):
    return asyncio.run(coro)


def test_hostname_resolving_to_internal_address_is_rejected():
    async def go():
        loop = asyncio.get_running_loop()
        with mock.patch.object(loop, "getaddrinfo",
                               _fake_getaddrinfo(["10.0.0.5"])):
            with pytest.raises(UnsafeURLError):
                await assert_safe_resolved("http://wiki.intern.example/")
    _run(go())


def test_any_internal_address_among_several_is_rejected():
    async def go():
        loop = asyncio.get_running_loop()
        with mock.patch.object(loop, "getaddrinfo",
                               _fake_getaddrinfo(["93.184.216.34", "127.0.0.1"])):
            with pytest.raises(UnsafeURLError):
                await assert_safe_resolved("http://mixed.example/")
    _run(go())


def test_public_resolution_passes():
    async def go():
        loop = asyncio.get_running_loop()
        with mock.patch.object(loop, "getaddrinfo",
                               _fake_getaddrinfo(["93.184.216.34"])):
            await assert_safe_resolved("http://public.example/")
    _run(go())


def test_allow_list_skips_resolution():
    async def go():
        loop = asyncio.get_running_loop()
        gai = mock.AsyncMock(side_effect=AssertionError("must not resolve"))
        with mock.patch.object(loop, "getaddrinfo", gai):
            await assert_safe_resolved(
                "http://wiki.intern.example/",
                parse_allowed_hosts(" Wiki.Intern.Example , other "),
            )
    _run(go())


def test_request_guard_checks_each_request():
    class _Req:
        def __init__(self, url):
            self.url = url

    async def go():
        guard = make_request_guard()
        with pytest.raises(UnsafeURLError):
            await guard(_Req("http://127.0.0.1/after-redirect"))
    _run(go())
