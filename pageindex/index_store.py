"""
Persistent global index over a folder of PageIndex tree JSON files.

This is the substrate for fast, cross-document retrieval (Phase 1, BM25-only).
It collapses the agentic per-doc tree walk into a single-step candidate
selection across *all* documents, and separates navigation (a small,
always-queryable manifest of titles/summaries) from content (node text fetched
by key on demand) so the retriever never re-parses every tree per query.

Storage is one SQLite file:
  - documents     : per-doc metadata (name, description, type, size, source)
  - nodes         : flattened tree nodes + text-by-key + breadcrumb/parent/depth
  - nodes_fts     : standalone FTS5 mirror for BM25 search across all docs
  - file_manifest : path -> (mtime, size, doc_id) for incremental rebuilds
  - index_meta    : schema version, build timestamp, (future) embedding model/dim

FTS5 ships with the stdlib sqlite3 on this platform, so BM25 needs no extra
dependency and no GPU server. Semantic embeddings are a Phase-2 add-on and are
intentionally not part of this module yet.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("pageindex.index_store")

SCHEMA_VERSION = "1"
DEFAULT_DB_NAME = ".pageindex_index.db"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ── Tree helpers ───────────────────────────────────────────────────────────────

def _iter_nodes_with_context(structure, parent_id=None, breadcrumb=(), depth=0):
    """Yield (node, parent_id, breadcrumb_titles, depth) for every node.

    breadcrumb_titles is the tuple of ancestor titles (root-first), used so a
    candidate from a global cross-doc search still carries the chapter/section
    context it sits under."""
    for node in structure or []:
        title = node.get("title", "") or ""
        yield node, parent_id, breadcrumb, depth
        children = node.get("nodes")
        if children:
            yield from _iter_nodes_with_context(
                children, node.get("node_id"), breadcrumb + (title,), depth + 1
            )


def _detect_type(structure) -> str:
    for node, *_ in _iter_nodes_with_context(structure):
        if "line_num" in node:
            return "md"
        if "start_index" in node or "end_index" in node:
            return "pdf"
    return "pdf"


def _doc_size(structure, doc_type: str) -> int:
    if doc_type == "pdf":
        return max((n.get("end_index") or 0 for n, *_ in _iter_nodes_with_context(structure)), default=0)
    return max((n.get("line_num") or 0 for n, *_ in _iter_nodes_with_context(structure)), default=0)


def _fts_match_query(question: str) -> str | None:
    """Build a safe FTS5 MATCH string from free text.

    We OR the word tokens (recall-oriented first stage; BM25 ranking sorts the
    noise down) and quote each so punctuation in the query can't break FTS5
    MATCH syntax. Returns None when there's nothing searchable."""
    tokens = [t for t in _WORD_RE.findall(question.lower()) if len(t) >= 2]
    if not tokens:
        return None
    # Dedupe preserving order; cap to keep the MATCH expression bounded.
    seen, ordered = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
        if len(ordered) >= 64:
            break
    return " OR ".join(f'"{t}"' for t in ordered)


# ── Index store ─────────────────────────────────────────────────────────────────

class IndexStore:
    """Build and query a persistent SQLite index over a folder of trees."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        # The openai-agents SDK runs sync function-tools (search_all_documents)
        # in a worker thread, so the connection is shared across threads. We
        # open with check_same_thread=False and serialize access with this lock;
        # tool calls are already serial (parallel_tool_calls=False) and reads are
        # idempotent, so contention is nil.
        self._lock = threading.RLock()

    # -- connection / schema --------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._ensure_schema(self._conn)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id           TEXT PRIMARY KEY,
                doc_name         TEXT,
                doc_description  TEXT,
                doc_type         TEXT,
                size             INTEGER,
                source_path      TEXT
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_rowid   INTEGER PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                node_id      TEXT,
                parent_id    TEXT,
                depth        INTEGER,
                title        TEXT,
                breadcrumb   TEXT,
                summary      TEXT,
                start_index  INTEGER,
                end_index    INTEGER,
                line_num     INTEGER,
                text         TEXT,
                text_len     INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_doc      ON nodes(doc_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_doc_node ON nodes(doc_id, node_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                title, breadcrumb, summary, text,
                doc_id UNINDEXED,
                node_rowid UNINDEXED
            );

            CREATE TABLE IF NOT EXISTS file_manifest (
                path    TEXT PRIMARY KEY,
                mtime   REAL,
                size    INTEGER,
                doc_id  TEXT
            );

            CREATE TABLE IF NOT EXISTS index_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO index_meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    # -- build ----------------------------------------------------------------

    def build(self, folder: str | Path, reindex: bool = False) -> dict:
        """Incrementally (re)build the index from a folder of *.json trees.

        Skips files whose (mtime, size) are unchanged since the last build
        unless ``reindex`` is set. Drops docs whose source file disappeared.
        Returns a summary dict."""
        folder = Path(folder).expanduser().resolve()
        conn = self.conn
        t0 = time.time()

        if reindex:
            conn.executescript(
                "DELETE FROM nodes; DELETE FROM nodes_fts; "
                "DELETE FROM documents; DELETE FROM file_manifest;"
            )
            conn.commit()

        prior = {
            row["path"]: (row["mtime"], row["size"], row["doc_id"])
            for row in conn.execute("SELECT path, mtime, size, doc_id FROM file_manifest")
        }

        seen_paths: set[str] = set()
        added = updated = skipped = 0
        node_total = 0

        for path in sorted(folder.glob("*.json")):
            if path.name == "_meta.json":
                continue
            path_str = str(path)
            seen_paths.add(path_str)
            st = path.stat()
            if not reindex and path_str in prior:
                pm, ps, _ = prior[path_str]
                if pm == st.st_mtime and ps == st.st_size:
                    skipped += 1
                    continue

            data = self._read_json(path)
            if data is None:
                continue
            if isinstance(data, list):
                structure, doc_name, doc_desc = data, path.stem, ""
            elif isinstance(data, dict):
                structure = data.get("structure") or []
                doc_name = data.get("doc_name") or path.stem
                doc_desc = data.get("doc_description") or ""
            else:
                logger.warning("skipping %s: unrecognized JSON shape", path.name)
                continue
            if not structure:
                logger.warning("skipping %s: no structure", path.name)
                continue

            doc_id = path.stem
            existing = path_str in prior
            self._delete_doc(conn, doc_id)
            n = self._insert_doc(conn, doc_id, doc_name, doc_desc, structure, path_str)
            conn.execute(
                "INSERT OR REPLACE INTO file_manifest(path, mtime, size, doc_id) VALUES(?,?,?,?)",
                (path_str, st.st_mtime, st.st_size, doc_id),
            )
            node_total += n
            updated += existing
            added += not existing
            logger.info("indexed %s (%d nodes)", doc_id, n)

        # Drop docs whose source file vanished.
        removed = 0
        for path_str, (_, _, doc_id) in prior.items():
            if path_str not in seen_paths:
                self._delete_doc(conn, doc_id)
                conn.execute("DELETE FROM file_manifest WHERE path=?", (path_str,))
                removed += 1

        conn.execute(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES('built_at', ?)",
            (str(time.time()),),
        )
        conn.commit()
        summary = {
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "removed": removed,
            "nodes_indexed": node_total,
            "elapsed_s": round(time.time() - t0, 3),
        }
        logger.info("index build: %s", summary)
        return summary

    @staticmethod
    def _read_json(path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("skipping %s: %s", path.name, e)
            return None

    @staticmethod
    def _delete_doc(conn: sqlite3.Connection, doc_id: str):
        conn.execute("DELETE FROM nodes_fts WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM nodes WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def _insert_doc(self, conn, doc_id, doc_name, doc_desc, structure, source_path) -> int:
        doc_type = _detect_type(structure)
        conn.execute(
            "INSERT INTO documents(doc_id, doc_name, doc_description, doc_type, size, source_path) "
            "VALUES(?,?,?,?,?,?)",
            (doc_id, doc_name, doc_desc, doc_type, _doc_size(structure, doc_type), source_path),
        )
        count = 0
        for node, parent_id, breadcrumb, depth in _iter_nodes_with_context(structure):
            title = node.get("title", "") or ""
            summary = node.get("summary", "") or ""
            text = node.get("text", "") or ""
            crumb = " > ".join(t for t in breadcrumb if t)
            cur = conn.execute(
                "INSERT INTO nodes(doc_id, node_id, parent_id, depth, title, breadcrumb, "
                "summary, start_index, end_index, line_num, text, text_len) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id, node.get("node_id"), parent_id, depth, title, crumb, summary,
                    node.get("start_index"), node.get("end_index"), node.get("line_num"),
                    text, len(text),
                ),
            )
            conn.execute(
                "INSERT INTO nodes_fts(rowid, title, breadcrumb, summary, text, doc_id, node_rowid) "
                "VALUES(?,?,?,?,?,?,?)",
                (cur.lastrowid, title, crumb, summary, text, doc_id, cur.lastrowid),
            )
            count += 1
        return count

    # -- query ----------------------------------------------------------------

    def is_empty(self) -> bool:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0

    def stats(self) -> dict:
        with self._lock:
            c = self.conn
            return {
                "documents": c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "nodes": c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                "db_path": str(self.db_path),
            }

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id, doc_name, doc_type, size FROM documents ORDER BY doc_id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT doc_id, doc_name, doc_description, doc_type, size, source_path "
                "FROM documents WHERE doc_id=?",
                (doc_id,),
            ).fetchone()
        return dict(row) if row else None

    def search(self, question: str, top_k: int = 50) -> list[dict]:
        """Global BM25 search across all documents.

        Returns ranked candidates (best first) as dicts with doc_id, node_id,
        title, breadcrumb, summary, span fields, text_len, and bm25 score
        (lower = better match, per SQLite's bm25())."""
        match = _fts_match_query(question)
        if not match:
            return []
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT n.doc_id, n.node_id, n.title, n.breadcrumb, n.summary,
                       n.start_index, n.end_index, n.line_num, n.text_len,
                       bm25(nodes_fts) AS score
                FROM nodes_fts
                JOIN nodes n ON n.node_rowid = nodes_fts.node_rowid
                WHERE nodes_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match, top_k),
            ).fetchall()
        return [dict(r) for r in rows]

    def _resolve_node_id(self, doc_id: str, node_id) -> str | None:
        """Resolve a node_id tolerating bare/unpadded ints (weak models drop
        quotes and zero-padding). Returns the canonical stored id or None."""
        if node_id is None or node_id == "":
            return None
        target = str(node_id).strip()
        if len(target) >= 2 and target[0] == target[-1] and target[0] in ("'", '"'):
            target = target[1:-1]
        with self._lock:
            row = self.conn.execute(
                "SELECT node_id FROM nodes WHERE doc_id=? AND node_id=?", (doc_id, target)
            ).fetchone()
            if row:
                return row["node_id"]
            if target.isdigit():
                widths = {
                    len(r["node_id"])
                    for r in self.conn.execute(
                        "SELECT DISTINCT node_id FROM nodes WHERE doc_id=? AND node_id IS NOT NULL",
                        (doc_id,),
                    )
                    if r["node_id"]
                }
                for w in sorted(widths):
                    if len(target) < w:
                        row = self.conn.execute(
                            "SELECT node_id FROM nodes WHERE doc_id=? AND node_id=?",
                            (doc_id, target.zfill(w)),
                        ).fetchone()
                        if row:
                            return row["node_id"]
        return None

    def get_node(self, doc_id: str, node_id) -> dict | None:
        resolved = self._resolve_node_id(doc_id, node_id)
        if resolved is None:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT doc_id, node_id, parent_id, depth, title, breadcrumb, summary, "
                "start_index, end_index, line_num, text, text_len "
                "FROM nodes WHERE doc_id=? AND node_id=?",
                (doc_id, resolved),
            ).fetchone()
        return dict(row) if row else None

    def get_node_content(self, doc_id: str, node_id) -> dict:
        """Fetch a node's content by key, falling back to summary when the tree
        was built without node text (config default if_add_node_text=no)."""
        node = self.get_node(doc_id, node_id)
        if node is None:
            return {"error": f"Node {node_id!r} not found in {doc_id}"}
        text = node.get("text") or ""
        if text:
            source = "text"
        elif node.get("summary"):
            text, source = node["summary"], "summary"
        else:
            text, source = "", "none"
        return {
            "doc_id": doc_id,
            "node_id": node.get("node_id"),
            "title": node.get("title", ""),
            "breadcrumb": node.get("breadcrumb", ""),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
            "line_num": node.get("line_num"),
            "source": source,
            "text": text,
        }
