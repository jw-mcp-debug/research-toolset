"""
Person directory connector — searches a local SQLite database of the
institution's people and their web pages (schema:
examples/person_directory/schema.sql).

Combination of:
  1. FTS5 (keyword search)
  2. vector similarity (semantic search)
  3. reranker (cross-encoder for the final order)
  4. hard person-name filter

Consent: only persons with `consent = 1` are ever returned. Content of
persons without consent is deleted from the searchable tables on first
use (see `ensure_consent_fresh`). Maintaining the consent flag is the
job of the institution-specific sync that fills the database.
"""

import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from src.core.tls import tls_verify

logger = logging.getLogger(__name__)


# ─── Person-name filter against fuzzy-match problems ───────────────────
#
# Problem: a directory query such as "Mira Okonkwo" also returns hits
# for other people called Mira — because FTS5 and the vector search
# match individual tokens and the reranker rates first-name
# similarity highly.
#
# Remedy: if the query looks like a person's name (2-3 words, each
# longer than 2 characters, no stop words), results are filtered hard
# after reranking: ALL name parts must occur in person_name
# (after diacritic and case normalisation).

_PERSON_NAME_STOPWORDS = {
    "dr", "dr.", "prof", "prof.", "herr", "frau", "herrn",
    "professor", "professorin", "doktor", "mister", "mr",
    "mrs", "ms", "mx", "von", "van", "de", "der", "la", "le",
}


def _normalize_name_for_match(s: str) -> str:
    """Normalise a name for comparison: remove diacritics, case-fold,
    strip punctuation.

    Examples:
        "Kovács"       → "kovacs"
        "Müller-Weber" → "muller weber"
        "Søren"        → "soren"
        "Dr. Brenner"  → "dr brenner"
    """
    if not s:
        return ""
    # NFKD decomposes umlauts/diacritics into base + combining characters
    decomposed = unicodedata.normalize("NFKD", s)
    # Remove combining characters (accents)
    without_accents = "".join(
        c for c in decomposed if not unicodedata.combining(c)
    )
    # lower-case
    lower = without_accents.lower()
    # Map non-NFKD characters (Scandinavian/Slavic).
    # These are letters of their own, not "base + accent".
    char_map = {
        "ø": "o", "œ": "oe", "æ": "ae", "ß": "ss",
        "đ": "d", "ð": "d", "þ": "th", "ł": "l",
        "ħ": "h", "ı": "i", "ĸ": "k",
    }
    for src, dst in char_map.items():
        if src in lower:
            lower = lower.replace(src, dst)
    # Replace non-letters with spaces (compound names)
    cleaned = re.sub(r"[^a-z0-9]+", " ", lower)
    return cleaned.strip()


def _extract_name_parts(query: str) -> list[str]:
    """Extract the substantial name parts from a query.

    Removes titles/honorifics (Dr., Prof. etc.) and particles (von, van).
    Returns a list of normalised name parts (at least 2 characters).
    """
    normalized = _normalize_name_for_match(query)
    parts = [
        w for w in normalized.split()
        if len(w) >= 2 and w not in _PERSON_NAME_STOPWORDS
    ]
    return parts


def _is_person_name_query(query: str) -> bool:
    """Heuristic: does the query look like a person's name?

    Criteria:
    - after removing titles/particles: 2-3 substantial words
    - no obvious non-person indicators (URLs, "site:", numbers, special
      characters such as ?, !)
    """
    if not query:
        return False

    # Non-person indicators
    non_person_markers = ("site:", "http://", "https://", "?", "!")
    if any(m in query.lower() for m in non_person_markers):
        return False

    parts = _extract_name_parts(query)
    return 2 <= len(parts) <= 3


def _name_matches_query(query: str, person_name: str) -> bool:
    """Check whether ALL name parts of the query occur in person_name
    (after diacritic and case normalisation).

    Examples:
        query="Mira Okonkwo", person_name="Mira Okonkwo"    → True
        query="Mira Okonkwo", person_name="Mira Lindqvist"  → False
        query="Okonkwo Mira", person_name="Mira Okonkwo"    → True (order irrelevant)
        query="Müller",       person_name="Anna Müller"     → True (single part ok)
    """
    query_parts = _extract_name_parts(query)
    if not query_parts:
        return True  # no parts to match → permissive

    pname_parts = set(_normalize_name_for_match(person_name).split())
    if not pname_parts:
        return False

    # ALL query parts must occur in person_name
    return all(qp in pname_parts for qp in query_parts)



@dataclass
class DirectoryPerson:
    """A search result from the person directory database."""
    pid: int
    person_name: str
    email: str
    org_path: str
    subject_area: str
    # Person-Details
    academic_title: str = ""
    status: str = ""
    homepage_url: str = ""
    # Position details (from person_orgs)
    role: str = ""
    phone: str = ""
    room: str = ""
    building: str = ""
    # Optional page information
    page_id: int | None = None
    page_title: str = ""
    page_url: str = ""
    page_content: str = ""
    page_type: str = ""
    # Scoring
    fts_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

    @property
    def real_url(self) -> str:
        """Return a real URL: homepage, page URL or the directory profile URL."""
        if self.homepage_url and self.homepage_url.startswith("http"):
            return self.homepage_url
        if self.page_url and self.page_url.startswith("http"):
            return self.page_url
        from src.institution import get_profile
        return get_profile().directory_url(self.pid)

    def to_markdown(self) -> str:
        """Format as Markdown for the research pipeline.

        IMPORTANT: output all available fields explicitly, so that the
        LLM does not hallucinate or omit details.

        The organisational hierarchy (org_path) is resolved:
        "Vice President for Research → Media Services" means "Media
        Services, belonging to the Vice President for Research" — the
        person works at the LAST element, not the first. The Markdown
        makes that unmistakable.
        """
        # Name WITH the correct title from the directory
        if self.academic_title:
            display_name = f"{self.academic_title} {self.person_name}"
        else:
            display_name = self.person_name

        lines = [f"## {display_name}"]
        lines.append(
            f"**Academic title (person directory):** "
            f"{self.academic_title if self.academic_title else '(no academic title recorded)'}"
        )
        if self.status:
            lines.append(f"**Staff status:** {self.status}")

        # ── Show role and subject area prominently ──
        # These are the person's ACTUAL roles.
        if self.role:
            lines.append(f"**Role:** {self.role}")
        if self.subject_area:
            lines.append(f"**Subject area:** {self.subject_area}")

        # ── Extract the organisational unit from the path ──
        # org_path has the form "A → B → C", where C is the person's direct
        # unit and A, B are parent units. Several paths may be separated by
        # "," (GROUP_CONCAT).
        if self.org_path:
            org_paths = [p.strip() for p in self.org_path.split(",")
                         if p.strip()]
            for org_p in org_paths:
                parts = [p.strip() for p in org_p.split("→")
                         if p.strip()]
                if len(parts) >= 2:
                    # last element = direct unit
                    # previous = parent hierarchy
                    direct_unit = parts[-1]
                    hierarchy = " → ".join(parts[:-1])
                    lines.append(
                        f"**Unit:** {direct_unit}"
                    )
                    lines.append(
                        f"**Organisation path:** {hierarchy}"
                    )
                elif len(parts) == 1:
                    lines.append(f"**Unit:** {parts[0]}")

        if self.email:
            lines.append(f"**E-mail:** {self.email}")
        if self.phone:
            lines.append(f"**Phone:** {self.phone}")
        if self.real_url:
            lines.append(f"**Profile:** {self.real_url}")
        if self.room and self.building:
            lines.append(f"**Location:** {self.building}, room {self.room}")
        elif self.room:
            lines.append(f"**Room:** {self.room}")
        elif self.building:
            lines.append(f"**Building:** {self.building}")
        if self.page_title:
            lines.append(f"\n### {self.page_title}")
        if self.page_content:
            lines.append(self.page_content[:5000])
        return "\n".join(lines)


@dataclass
class DirectorySearchConfig:
    db_path: str = ""
    embedder_base_url: str = ""
    embedder_model: str = ""
    embedder_api_key: str = ""
    reranker_base_url: str = ""
    reranker_model: str = ""
    reranker_api_key: str = ""
    # Search
    fts_limit: int = 50
    vector_limit: int = 50
    rerank_top_n: int = 20
    final_limit: int = 10

    @property
    def enabled(self) -> bool:
        return bool(self.db_path) and Path(self.db_path).exists()


class PersonDirectoryConnector:
    """Search the local person directory SQLite database."""

    def __init__(self, config: DirectorySearchConfig):
        self.config = config
        self._conn: sqlite3.Connection | None = None
        self._embeddings_cache: np.ndarray | None = None
        self._embeddings_page_ids: list[int] = []
        self._consent_checked = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def ensure_consent_fresh(self, max_age_days: int = 7):
        """Remove everything searchable about persons without consent.

        Consent itself is maintained by the job that fills the database
        (it sets `persons.consent`). Here, once per process, pages,
        embeddings and index entries of persons with consent = 0 are
        deleted and the full-text index is rebuilt, so a revoked consent
        takes effect even if the sync job left content behind.
        `max_age_days` is accepted for interface compatibility.
        """
        if self._consent_checked or not self.enabled:
            return
        conn = self._get_conn()
        stale = conn.execute("""
            SELECT p.pid FROM persons p
            WHERE p.consent = 0 AND (
                EXISTS (SELECT 1 FROM pages WHERE pid = p.pid)
                OR EXISTS (SELECT 1 FROM search_index WHERE pid = p.pid)
            )
        """).fetchall()
        if stale:
            logger.warning(
                "Person directory: %d person(s) without consent still had "
                "searchable data — deleting", len(stale),
            )
            for row in stale:
                pid = row[0]
                conn.execute("DELETE FROM embeddings WHERE page_id IN "
                             "(SELECT page_id FROM pages WHERE pid=?)", (pid,))
                conn.execute("DELETE FROM search_index WHERE pid=?", (pid,))
                conn.execute("DELETE FROM pages WHERE pid=?", (pid,))
            conn.commit()
            self._rebuild_fts_consent(conn)
            self._embeddings_cache = None
        self._consent_checked = True

    @staticmethod
    def _rebuild_fts_consent(conn: sqlite3.Connection):
        """Rebuild the FTS index (only persons with consent)."""
        conn.execute("DELETE FROM search_index")

        rows = conn.execute("""
            SELECT p.pid, p.given_name || ' ' || p.family_name AS person_name,
                   p.email, p.academic_title, p.orcid,
                   GROUP_CONCAT(DISTINCT o.full_path) AS org_path,
                   GROUP_CONCAT(DISTINCT po.subject_area) AS subject_area,
                   GROUP_CONCAT(DISTINCT po.role) AS role
            FROM persons p
            LEFT JOIN person_orgs po ON p.pid = po.pid
            LEFT JOIN orgs o ON po.org_id = o.org_id
            WHERE p.consent = 1
            GROUP BY p.pid
        """).fetchall()

        for row in rows:
            person_text = " ".join(filter(None, [
                row["person_name"], row["academic_title"], row["email"],
                row["org_path"], row["subject_area"], row["role"],
                row["orcid"],
            ]))
            conn.execute("""
                INSERT INTO search_index (pid, page_id, person_name,
                                          org_path, subject_area, title, content)
                VALUES (?, NULL, ?, ?, ?, '', ?)
            """, (row["pid"], row["person_name"] or "",
                  row["org_path"] or "", row["subject_area"] or "",
                  person_text))

        rows = conn.execute("""
            SELECT pg.page_id, pg.pid, pg.title, pg.content,
                   p.given_name || ' ' || p.family_name AS person_name,
                   GROUP_CONCAT(DISTINCT o.full_path) AS org_path,
                   GROUP_CONCAT(DISTINCT po.subject_area) AS subject_area
            FROM pages pg
            LEFT JOIN persons p ON pg.pid = p.pid
            LEFT JOIN person_orgs po ON pg.pid = po.pid
            LEFT JOIN orgs o ON po.org_id = o.org_id
            WHERE p.consent = 1 OR pg.pid IS NULL
            GROUP BY pg.page_id
        """).fetchall()

        for row in rows:
            conn.execute("""
                INSERT INTO search_index (pid, page_id, person_name,
                                          org_path, subject_area, title, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (row["pid"], row["page_id"],
                  row["person_name"] or "", row["org_path"] or "",
                  row["subject_area"] or "", row["title"] or "",
                  (row["content"] or "")[:50000]))

        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not Path(self.config.db_path).exists():
                raise FileNotFoundError(
                    f"Person directory database not found: {self.config.db_path}"
                )
            self._conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            # Databases without a consent column: add it (0 = not released)
            try:
                self._conn.execute("SELECT consent FROM persons LIMIT 1")
            except Exception:
                logger.warning(
                    "Person directory DB without a consent column — "
                    "all persons count as not released until the next sync"
                )
                self._conn.execute(
                    "ALTER TABLE persons ADD COLUMN "
                    "consent INTEGER DEFAULT 0"
                )
                self._conn.commit()
        return self._conn

    async def search(self, query: str, max_results: int = 10,
                     subject_area: str = "", org_filter: str = "",
                     query_anchor=None,
                     ) -> list[DirectoryPerson]:
        """Hybrid search: FTS5 + vector + reranker.

        Optionally faceted by subject_area or org_path.

        Args:
            query: free-text search query.
            max_results: maximum number of results (default 10).
            subject_area: optional subject-area filter.
            org_filter: optional organisation filter.
            query_anchor: optional `QueryAnchor` from the pipeline. If
                          present, `is_person()` is taken from it instead
                          of from the heuristic, which is not reliable
                          (e.g. "Preis-Leistungs-Verhältnis" would be
                          taken for a person).
        """
        results: dict[str, DirectoryPerson] = {}  # key: "pid-page_id"

        # Phase 1: FTS5 search
        fts_results = self._fts_search(query)
        for r in fts_results:
            key = f"{r.pid}-{r.page_id}"
            r.fts_score = 1.0
            results[key] = r

        # Phase 1b: faceted filters (if given)
        if subject_area or org_filter:
            filtered = {}
            sg_lower = subject_area.lower() if subject_area else ""
            org_lower = org_filter.lower() if org_filter else ""
            for key, r in results.items():
                r_sg = (r.subject_area or "").lower() if hasattr(r, 'subject_area') else ""
                r_org = (r.org_path or "").lower()
                if sg_lower and sg_lower not in r_sg:
                    continue
                if org_lower and org_lower not in r_org:
                    continue
                filtered[key] = r
            # If the filter removes everything, fall back to unfiltered
            if filtered:
                results = filtered
                logger.debug(
                    f"Person directory faceted: {len(results)} of {len(fts_results)} "
                    f"by subject_area='{subject_area}', org='{org_filter}'"
                )

        # Phase 2: vector search (if an embedder is available)
        if self.config.embedder_api_key:
            try:
                vector_results = await self._vector_search(query)
                for r in vector_results:
                    key = f"{r.pid}-{r.page_id}"
                    if key in results:
                        results[key].vector_score = r.vector_score
                    else:
                        results[key] = r
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        if not results:
            return []

        # Phase 3: reranker (if available)
        all_results = list(results.values())

        if self.config.reranker_api_key and len(all_results) > 1:
            try:
                all_results = await self._rerank(query, all_results)
            except Exception as e:
                logger.warning(f"Reranker failed: {e}")
                # Fallback: sort by the combined score
                for r in all_results:
                    r.final_score = r.fts_score * 0.4 + r.vector_score * 0.6
                all_results.sort(key=lambda r: r.final_score, reverse=True)
        else:
            # Without a reranker: weighted score
            for r in all_results:
                r.final_score = r.fts_score * 0.4 + r.vector_score * 0.6
            all_results.sort(key=lambda r: r.final_score, reverse=True)

        # Phase 4: hard person-name filter
        # If the query looks like a person's name, ALL name parts must occur
        # in person_name — otherwise the hit is a fuzzy-match error and is
        # discarded. query_anchor (from the pipeline classifier) takes
        # precedence over the local heuristic — otherwise the heuristic would
        # also classify "Preis-Leistungs-Verhältnis" as a person.
        if query_anchor is not None:
            is_person = bool(query_anchor.is_person())
        else:
            is_person = _is_person_name_query(query)

        if is_person:
            pre_count = len(all_results)
            all_results = [
                r for r in all_results
                if _name_matches_query(query, r.person_name)
            ]
            post_count = len(all_results)
            if pre_count != post_count:
                logger.info(
                    f"Person directory name filter: {pre_count} → {post_count} "
                    f"hits (query='{query}')"
                )

        return all_results[:max_results]

    def _fts_search(self, query: str) -> list[DirectoryPerson]:
        """FTS5 full-text search."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT si.pid, si.page_id, si.person_name, si.org_path,
                       si.subject_area, si.title,
                       snippet(search_index, 6, '<b>', '</b>', '...', 40) AS snippet,
                       rank
                FROM search_index si
                WHERE search_index MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, self.config.fts_limit)).fetchall()
        except sqlite3.OperationalError as e:
            # FTS query error (e.g. special characters) — fall back to a simple search
            logger.debug(f"FTS error, fallback: {e}")
            # Simple word search
            words = query.split()
            fts_query = " OR ".join(f'"{w}"' for w in words if len(w) > 1)
            if not fts_query:
                return []
            try:
                rows = conn.execute("""
                    SELECT si.pid, si.page_id, si.person_name, si.org_path,
                           si.subject_area, si.title,
                           snippet(search_index, 6, '<b>', '</b>', '...', 40) AS snippet,
                           rank
                    FROM search_index si
                    WHERE search_index MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, self.config.fts_limit)).fetchall()
            except Exception:
                return []

        results = []
        for row in rows:
            pid = row["pid"]
            # Load person details (only with consent)
            person = conn.execute(
                "SELECT email, homepage_url, academic_title, status FROM persons "
                "WHERE pid=? AND consent = 1",
                (pid,)
            ).fetchone()

            # Without consent: skip the entry
            if pid and not person:
                continue

            # Load position details (role, phone, room, building)
            role = ""
            phone = ""
            room = ""
            building = ""
            if pid:
                orgs = conn.execute("""
                    SELECT role, phone, room, building
                    FROM person_orgs WHERE pid=?
                """, (pid,)).fetchall()
                # take the first non-empty values
                for org in orgs:
                    if not role and org["role"]:
                        role = org["role"]
                    if not phone and org["phone"]:
                        phone = org["phone"]
                    if not room and org["room"]:
                        room = org["room"]
                    if not building and org["building"]:
                        building = org["building"]

            page_content = ""
            page_url = ""
            page_type = ""
            if row["page_id"]:
                page = conn.execute(
                    "SELECT url, content, page_type FROM pages WHERE page_id=?",
                    (row["page_id"],)
                ).fetchone()
                if page:
                    page_content = page["content"] or ""
                    page_url = page["url"] or ""
                    page_type = page["page_type"] or ""

            results.append(DirectoryPerson(
                pid=pid,
                person_name=row["person_name"] or "",
                email=person["email"] if person else "",
                org_path=row["org_path"] or "",
                subject_area=row["subject_area"] or "",
                academic_title=person["academic_title"] if person else "",
                status=person["status"] if person else "",
                homepage_url=person["homepage_url"] if person else "",
                role=role,
                phone=phone,
                room=room,
                building=building,
                page_id=row["page_id"],
                page_title=row["title"] or "",
                page_url=page_url,
                page_content=page_content,
                page_type=page_type,
                fts_score=abs(row["rank"]) if row["rank"] else 0,
            ))

        # Normalise FTS scores (highest = 1.0)
        if results:
            max_score = max(r.fts_score for r in results) or 1.0
            for r in results:
                r.fts_score = r.fts_score / max_score

        return results

    async def _vector_search(self, query: str) -> list[DirectoryPerson]:
        """Semantic search over embeddings."""
        import httpx

        # Embed the query
        async with httpx.AsyncClient(timeout=30.0, verify=tls_verify()) as client:
            resp = await client.post(
                f"{self.config.embedder_base_url}/embeddings",
                json={
                    "model": self.config.embedder_model,
                    "input": [query],
                },
                headers={
                    "Authorization": f"Bearer {self.config.embedder_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            query_emb = np.array(
                resp.json()["data"][0]["embedding"], dtype=np.float32
            )

        # Load all embeddings (cached)
        if self._embeddings_cache is None:
            self._load_embeddings_cache()

        if self._embeddings_cache is None or len(self._embeddings_cache) == 0:
            return []

        # Cosine Similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = self._embeddings_cache @ query_norm

        # Top-N
        top_indices = np.argsort(scores)[::-1][:self.config.vector_limit]

        conn = self._get_conn()
        results = []
        for idx in top_indices:
            if scores[idx] < 0.1:  # Threshold
                break

            page_id = self._embeddings_page_ids[idx]
            row = conn.execute("""
                SELECT p.page_id, p.pid, p.title, p.url, p.content, p.page_type,
                       per.given_name || ' ' || per.family_name AS person_name,
                       per.email, per.academic_title, per.status, per.homepage_url,
                       GROUP_CONCAT(DISTINCT o.full_path) AS org_path,
                       GROUP_CONCAT(DISTINCT po.subject_area) AS subject_area
                FROM pages p
                LEFT JOIN persons per ON p.pid = per.pid
                LEFT JOIN person_orgs po ON p.pid = po.pid
                LEFT JOIN orgs o ON po.org_id = o.org_id
                WHERE p.page_id = ?
                AND (per.consent = 1 OR p.pid IS NULL)
                GROUP BY p.page_id
            """, (page_id,)).fetchone()

            if row:
                # Load position details for consistency
                pid = row["pid"] or 0
                role = ""
                phone = ""
                room = ""
                building = ""
                if pid:
                    orgs = conn.execute("""
                        SELECT role, phone, room, building
                        FROM person_orgs WHERE pid=?
                    """, (pid,)).fetchall()
                    for org in orgs:
                        if not role and org["role"]:
                            role = org["role"]
                        if not phone and org["phone"]:
                            phone = org["phone"]
                        if not room and org["room"]:
                            room = org["room"]
                        if not building and org["building"]:
                            building = org["building"]

                results.append(DirectoryPerson(
                    pid=pid,
                    person_name=row["person_name"] or "",
                    email=row["email"] or "",
                    org_path=row["org_path"] or "",
                    subject_area=row["subject_area"] or "",
                    academic_title=row["academic_title"] or "",
                    status=row["status"] or "",
                    homepage_url=row["homepage_url"] or "",
                    role=role,
                    phone=phone,
                    room=room,
                    building=building,
                    page_id=page_id,
                    page_title=row["title"] or "",
                    page_url=row["url"] or "",
                    page_content=row["content"] or "",
                    page_type=row["page_type"] or "",
                    vector_score=float(scores[idx]),
                ))

        return results

    def _load_embeddings_cache(self):
        """Load embeddings into memory (only with consent)."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT e.page_id, e.embedding
            FROM embeddings e
            JOIN pages p ON e.page_id = p.page_id
            LEFT JOIN persons per ON p.pid = per.pid
            WHERE per.consent = 1 OR p.pid IS NULL
        """).fetchall()

        if not rows:
            self._embeddings_cache = np.array([])
            self._embeddings_page_ids = []
            return

        page_ids = []
        embeddings = []
        for row in rows:
            page_ids.append(row["page_id"])
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings.append(emb)

        matrix = np.stack(embeddings)
        # Normalise for fast cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        self._embeddings_cache = matrix / norms
        self._embeddings_page_ids = page_ids

        logger.info(
            f"Person directory embeddings loaded: {len(page_ids)} vectors, "
            f"dimension {matrix.shape[1]}"
        )

    async def _rerank(self, query: str,
                      results: list[DirectoryPerson]) -> list[DirectoryPerson]:
        """Rerank the results with the cross-encoder."""
        import httpx

        # Take the top N for reranking
        candidates = results[:self.config.rerank_top_n]

        documents = []
        for r in candidates:
            text = f"{r.person_name}. {r.org_path}. {r.subject_area}. "
            if r.page_title:
                text += f"{r.page_title}. "
            if r.page_content:
                text += r.page_content[:1000]
            documents.append(text)

        async with httpx.AsyncClient(timeout=30.0, verify=tls_verify()) as client:
            resp = await client.post(
                self.config.reranker_base_url,
                json={
                    "model": self.config.reranker_model,
                    "query": query,
                    "documents": documents,
                },
                headers={
                    "Authorization": f"Bearer {self.config.reranker_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Assign the scores
        for item in data.get("results", []):
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if idx < len(candidates):
                candidates[idx].rerank_score = score
                candidates[idx].final_score = score

        # Sort by rerank score
        candidates.sort(key=lambda r: r.final_score, reverse=True)
        return candidates

    def get_person_details(self, pid: int) -> dict | None:
        """Load full person details (only with consent)."""
        conn = self._get_conn()
        person = conn.execute(
            "SELECT * FROM persons WHERE pid=? AND consent = 1", (pid,)
        ).fetchone()
        if not person:
            return None

        orgs = conn.execute("""
            SELECT po.*, o.full_path, o.name AS org_name
            FROM person_orgs po
            LEFT JOIN orgs o ON po.org_id = o.org_id
            WHERE po.pid = ?
        """, (pid,)).fetchall()

        pages = conn.execute(
            "SELECT * FROM pages WHERE pid=?", (pid,)
        ).fetchall()

        return {
            "person": dict(person),
            "orgs": [dict(o) for o in orgs],
            "pages": [dict(p) for p in pages],
        }

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Verification ─────────────────────────────────────

    def verify_person(self, name: str) -> dict | None:
        """Check whether a person exists in the person directory.

        Returns the person's details if found, otherwise None.
        Searches by family name + optional given name.
        """
        conn = self._get_conn()
        parts = name.strip().split()
        if not parts:
            return None

        # Remove academic titles
        title_prefixes = {
            "prof.", "prof", "dr.", "dr", "pd", "jun.-prof.",
            "jun.-prof", "em.", "dipl.-ing.", "dipl.", "ing.",
        }
        clean_parts = [
            p for p in parts
            if p.lower().rstrip(".") not in title_prefixes
            and p.lower() not in title_prefixes
        ]
        if not clean_parts:
            return None

        # family name = last word, given name = the rest
        family_name = clean_parts[-1]
        given_name_prefix = " ".join(clean_parts[:-1]) if len(clean_parts) > 1 else ""

        # Exact search (only with consent)
        if given_name_prefix:
            row = conn.execute(
                "SELECT pid, given_name, family_name, academic_title, status, homepage_url "
                "FROM persons WHERE family_name = ? AND given_name LIKE ? AND consent = 1",
                (family_name, f"{given_name_prefix}%")
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT pid, given_name, family_name, academic_title, status, homepage_url "
                "FROM persons WHERE family_name = ? AND consent = 1",
                (family_name,)
            ).fetchone()

        if row:
            # Load the units
            orgs = conn.execute("""
                SELECT o.full_path, po.role, po.subject_area
                FROM person_orgs po
                LEFT JOIN orgs o ON po.org_id = o.org_id
                WHERE po.pid = ?
            """, (row["pid"],)).fetchall()

            return {
                "pid": row["pid"],
                "given_name": row["given_name"],
                "name": row["name"],
                "academic_title": row["academic_title"] or "",
                "status": row["status"] or "",
                "homepage_url": row["homepage_url"] or "",
                "orgs": [dict(o) for o in orgs],
                "found": True,
            }

        # Fuzzy: family name only (only with consent)
        if given_name_prefix:
            rows = conn.execute(
                "SELECT pid, given_name, family_name, academic_title, status FROM persons "
                "WHERE family_name = ? AND consent = 1 LIMIT 5",
                (family_name,)
            ).fetchall()
            if rows:
                return {
                    "found": False,
                    "similar": [
                        f"{r['given_name']} {r['name']} ({r['academic_title'] or 'kein Titel'}, {r['status'] or '?'})"
                        for r in rows
                    ],
                    "note": f"'{name}' not found exactly, but {len(rows)} person(s) with the family name '{family_name}'",
                }

        return None

    def verify_persons_batch(self, names: list[str]) -> dict[str, dict | None]:
        """Check several person names against the directory.

        Returns: {name: result_or_None}
        """
        return {name: self.verify_person(name) for name in names}
