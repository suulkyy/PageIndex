"""
Build / refresh the persistent SQLite index over a folder of PageIndex trees.

    python3 build_index.py --folder ./results
    python3 build_index.py --folder ./results --reindex        # force full rebuild
    python3 build_index.py --folder ./results --db ./my_index.db

The index powers the fast / hybrid retrieval modes in retrieve_pageindex.py.
Rebuilds are incremental: unchanged files (same mtime + size) are skipped, so
re-running after adding a few trees is cheap. The default DB path lives inside
the indexed folder as ``.pageindex_index.db``.

retrieve_pageindex.py auto-refreshes this index at startup too, so an explicit
build is optional — it's here for batch/offline indexing and inspection.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pageindex.index_store import DEFAULT_DB_NAME, IndexStore


def default_db_path(folder: Path) -> Path:
    return folder / DEFAULT_DB_NAME


def main():
    parser = argparse.ArgumentParser(
        description="Build/refresh the SQLite index over a folder of PageIndex tree JSON files."
    )
    parser.add_argument("--folder", default="./results",
                        help="Folder containing PageIndex tree JSON files (default: ./results).")
    parser.add_argument("--db", default=None,
                        help="Index DB path (default: <folder>/.pageindex_index.db).")
    parser.add_argument("--reindex", action="store_true",
                        help="Force a full rebuild instead of an incremental refresh.")
    parser.add_argument("--embeddings", action="store_true",
                        help="Also compute + store node embeddings (Phase 2 semantic recall). "
                             "Requires the embedding server (default :8001). Incremental unless --reembed.")
    parser.add_argument("--reembed", action="store_true",
                        help="Re-embed all nodes (use after changing the embedding model).")
    parser.add_argument("--embed-base-url", default=None,
                        help="Embedding server base URL (default env PAGEINDEX_EMBED_BASE_URL or http://localhost:8001/v1).")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model name (default env PAGEINDEX_EMBED_MODEL or rag-embed).")
    parser.add_argument("--embed-body-chars", type=int, default=1000,
                        help="Chars of node body text appended to the embedded string (A1 semantic "
                             "recall). Default 1000 (validated sweet spot); 0 = metadata only "
                             "(title+breadcrumb+summary). Changing this requires --reembed.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log per-document indexing progress.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    db_path = Path(args.db).expanduser().resolve() if args.db else default_db_path(folder)

    print(f"Indexing {folder} -> {db_path}")
    with IndexStore(db_path) as store:
        summary = store.build(folder, reindex=args.reindex)
        print(
            "Done: {added} added, {updated} updated, {skipped} skipped, "
            "{removed} removed | {nodes_indexed} nodes touched in {elapsed_s}s".format(**summary)
        )

        if args.embeddings or args.reembed:
            from pageindex import semantic
            cfg = semantic.embed_config(base_url=args.embed_base_url, model=args.embed_model)
            print(f"Embedding nodes via {cfg['base_url']} (model={cfg['model']}, batch={cfg['batch']}) ...")

            def embed_fn(texts):
                return semantic.embed_texts(
                    texts, base_url=args.embed_base_url, model=args.embed_model,
                )

            emb = store.build_embeddings(embed_fn, reembed=args.reembed, model_name=cfg["model"],
                                         body_chars=args.embed_body_chars)
            print(
                "Embeddings: {embedded} new, {total} total, dim={dim}".format(**emb)
                + f", body_chars={args.embed_body_chars}"
            )

        stats = store.stats()

    print(f"Index now holds {stats['documents']} document(s), {stats['nodes']} node(s).")


if __name__ == "__main__":
    main()
