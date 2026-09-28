"""
Local files connector — for uploaded context documents.
"""

import logging
import os
import time
from pathlib import Path

from src.connectors.base import BaseConnector
from src.pipeline.models import SearchResult, SourceDocument, SourceType

logger = logging.getLogger(__name__)


class LocalFileConnector(BaseConnector):
    """Connector for local files (uploads and configured paths)."""

    name = "local_files"

    def __init__(self, allowed_paths: list[str] | None = None):
        self.allowed_paths = allowed_paths or []

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search local directories by file name."""
        results = []
        query_lower = query.lower()

        for base_path in self.allowed_paths:
            if not os.path.isdir(base_path):
                continue
            for root, dirs, files in os.walk(base_path):
                for fname in files:
                    if query_lower in fname.lower():
                        full_path = os.path.join(root, fname)
                        results.append(SearchResult(
                            title=fname,
                            url=f"file://{full_path}",
                            snippet=f"local file in {root}",
                            source_type=SourceType.LOCAL_FILE,
                            connector_name=self.name,
                        ))
                        if len(results) >= max_results:
                            return results
        return results

    async def fetch(self, url: str) -> SourceDocument:
        """Read a local file."""
        t0 = time.monotonic()

        # Remove the file:// prefix
        path = url.replace("file://", "")

        if not os.path.exists(path):
            return SourceDocument(
                source_type=SourceType.LOCAL_FILE,
                url=url, title=os.path.basename(path),
                content=f"[file not found: {path}]",
            )

        try:
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    content = Path(path).read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                content = f"[encoding not recognised for: {path}]"

            return SourceDocument(
                source_type=SourceType.LOCAL_FILE,
                url=url,
                title=os.path.basename(path),
                content=content,
                fetch_time_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            return SourceDocument(
                source_type=SourceType.LOCAL_FILE,
                url=url, title=os.path.basename(path),
                content=f"[error: {e}]",
                fetch_time_seconds=time.monotonic() - t0,
            )

    def can_handle(self, url: str) -> bool:
        return url.startswith("file://")
