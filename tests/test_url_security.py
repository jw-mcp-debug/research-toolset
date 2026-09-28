"""
Tests for `src.core.url_security` (SSRF protection).

Acceptance: all common SSRF vectors are blocked:
  - localhost / 127.0.0.1 / ::1
  - RFC 1918 private addresses
  - link-local (169.254.x.x — cloud metadata)
  - file:// and other non-HTTP schemes
  - empty/malformed URLs
Public URLs are let through.
"""

import unittest

from src.core.url_security import (
    UnsafeURLError,
    assert_safe_url,
    is_safe_public_url,
)


class TestPublicUrlsAllowed(unittest.TestCase):

    def test_https_public(self):
        ok, reason = is_safe_public_url("https://example.com/path")
        self.assertTrue(ok, reason)

    def test_http_public(self):
        ok, _ = is_safe_public_url("http://example.com")
        self.assertTrue(ok)

    def test_public_ip_explicit(self):
        ok, _ = is_safe_public_url("https://8.8.8.8/")
        self.assertTrue(ok)

    def test_url_with_port(self):
        ok, _ = is_safe_public_url("https://example.com:8080/")
        self.assertTrue(ok)


class TestLocalhostBlocked(unittest.TestCase):

    def test_localhost_name(self):
        ok, reason = is_safe_public_url("http://localhost/")
        self.assertFalse(ok)
        self.assertIn("ocalhost", reason)

    def test_127_0_0_1(self):
        ok, reason = is_safe_public_url("http://127.0.0.1/")
        self.assertFalse(ok)

    def test_ipv6_loopback(self):
        ok, reason = is_safe_public_url("http://[::1]/")
        self.assertFalse(ok)

    def test_localhost_with_port(self):
        ok, _ = is_safe_public_url("http://localhost:7860/admin")
        self.assertFalse(ok)


class TestPrivateIPsBlocked(unittest.TestCase):

    def test_rfc1918_10(self):
        ok, reason = is_safe_public_url("http://10.0.0.5/secret")
        self.assertFalse(ok)
        self.assertIn("rivate", reason)

    def test_rfc1918_192_168(self):
        ok, _ = is_safe_public_url("http://192.168.1.1/")
        self.assertFalse(ok)

    def test_rfc1918_172_16(self):
        ok, _ = is_safe_public_url("http://172.16.0.1/")
        self.assertFalse(ok)

    def test_172_31_at_boundary(self):
        ok, _ = is_safe_public_url("http://172.31.255.255/")
        self.assertFalse(ok)

    def test_172_32_is_public(self):
        """172.32.x.x is outside 172.16/12 — public."""
        ok, _ = is_safe_public_url("http://172.32.0.1/")
        self.assertTrue(ok)


class TestLinkLocalBlocked(unittest.TestCase):

    def test_aws_metadata(self):
        """169.254.169.254 is the cloud metadata endpoint — the classic SSRF target.
        (Python's ipaddress classifies it as is_private, which is enough
        for blocking — whether declared 'private' or 'link-local'.)"""
        ok, reason = is_safe_public_url("http://169.254.169.254/")
        self.assertFalse(ok)
        # 'rivate' or 'ink-local' — both trigger the block
        self.assertTrue("rivate" in reason or "ink-local" in reason,
                        f"unexpected reason: {reason!r}")


class TestNonHttpSchemes(unittest.TestCase):

    def test_file_scheme(self):
        ok, reason = is_safe_public_url("file:///etc/passwd")
        self.assertFalse(ok)
        self.assertIn("cheme", reason)

    def test_ftp_scheme(self):
        ok, _ = is_safe_public_url("ftp://example.com/")
        self.assertFalse(ok)

    def test_gopher_scheme(self):
        ok, _ = is_safe_public_url("gopher://example.com/")
        self.assertFalse(ok)

    def test_javascript_scheme(self):
        ok, _ = is_safe_public_url("javascript:alert(1)")
        self.assertFalse(ok)


class TestMalformedURLs(unittest.TestCase):

    def test_empty(self):
        self.assertFalse(is_safe_public_url("")[0])
        self.assertFalse(is_safe_public_url("   ")[0])

    def test_no_scheme(self):
        ok, _ = is_safe_public_url("example.com/path")
        self.assertFalse(ok)

    def test_no_host(self):
        ok, _ = is_safe_public_url("http://")
        self.assertFalse(ok)

    def test_none(self):
        ok, _ = is_safe_public_url(None)
        self.assertFalse(ok)


class TestWhitelist(unittest.TestCase):

    def test_internal_host_via_whitelist(self):
        """Internal wiki allowed if on the allow list."""
        ok, _ = is_safe_public_url(
            "http://wiki.intern.example.com/",
            allowed_internal_hosts={"wiki.intern.example.com"},
        )
        self.assertTrue(ok)


class TestAssertSafeUrl(unittest.TestCase):

    def test_raises_on_unsafe(self):
        with self.assertRaises(UnsafeURLError):
            assert_safe_url("http://localhost/")

    def test_silent_on_safe(self):
        # no raise
        assert_safe_url("https://example.com/")

    def test_exception_has_url_and_reason(self):
        try:
            assert_safe_url("http://10.0.0.5/")
        except UnsafeURLError as e:
            self.assertEqual(e.url, "http://10.0.0.5/")
            self.assertIn("rivate", e.reason)


if __name__ == "__main__":
    unittest.main()
