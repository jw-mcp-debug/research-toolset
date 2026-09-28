"""TLS certificate verification for outgoing HTTP requests.

Verification is on by default. For endpoints with certificates from an
internal CA, point SSL_CERT_FILE (or SSL_CERT_DIR) at the CA bundle;
httpx honours both. TLS_VERIFY=false switches verification off entirely
and is meant only as a last resort on trusted networks.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_warned = False


def tls_verify() -> bool:
    """Return the `verify` value for httpx clients."""
    global _warned
    raw = os.environ.get("TLS_VERIFY", "true").strip().lower()
    verify = raw not in ("false", "0", "no", "off")
    if not verify and not _warned:
        logger.warning(
            "TLS_VERIFY=false: certificate verification is DISABLED for "
            "embedder, reranker, person directory and Solr requests"
        )
        _warned = True
    return verify
