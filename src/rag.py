"""RAG layer: ChromaDB + sentence-transformers (CPU), lazy-loaded.

Two corpora:
  - `knowledge`: curated, vetted technique cards (works from session one).
  - `history`:   the SAME user's past assessments only (per-user similarity).
Heavy dependencies are imported inside functions so `import src.rag` stays
light and the import smoke test needs no network or GPU.
"""
import re
from pathlib import Path

from .config import CHROMA_DIR, KNOWLEDGE_DIR

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_client = None
_embed_fn = None


def _require_deps():
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "RAG needs extra packages: run `pip install -r requirements.txt` "
            f"(missing: {e.name})."
        )


def _get_client():
    global _client
    if _client is None:
        _require_deps()
        import chromadb
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _require_deps()
        import chromadb.utils.embedding_functions as ef
        _embed_fn = ef.SentenceTransformerEmbeddingFunction(
            model_name=_EMBED_MODEL_NAME, device="cpu"
        )
    return _embed_fn


def parse_card(path: Path) -> dict:
    """Parse a knowledge/*.md card into labeled sections."""
    text = path.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    current, buf = None, []
    for line in text.splitlines():
        m = re.match(r"^([A-Z_]+):\s*(.*)$", line)
        if m:
            if current:
                fields[current] = "\n".join(buf).strip()
            current, buf = m.group(1), [m.group(2)]
        elif current:
            buf.append(line)
    if current:
        fields[current] = "\n".join(buf).strip()
    fields["_path"] = str(path)
    return fields


def build_index(force: bool = False) -> int:
    """Ingest knowledge/*.md cards into the `knowledge` collection.

    Returns the number of cards indexed. First run downloads the embedding
    model (~90MB) — needs internet once.
    """
    client = _get_client()
    try:
        client.delete_collection("knowledge")
    except Exception:
        pass
    col = client.get_or_create_collection(
        "knowledge", embedding_function=_get_embed_fn()
    )
    ids, docs, metas = [], [], []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        card = parse_card(md)
        cid = card.get("ID", md.stem)
        body = (
            f"TITLE: {card.get('TITLE', '')}\n"
            f"WHEN_TO_USE: {card.get('WHEN_TO_USE', '')}\n"
            f"STEPS:\n{card.get('STEPS', '')}"
        )
        ids.append(cid)
        docs.append(body)
        metas.append({
            "title": card.get("TITLE", ""),
            "source": card.get("SOURCE", ""),
            "license": card.get("LICENSE", ""),
        })
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def index_user_history(user_id: int, assessments: list[dict]) -> int:
    """Index THIS user's past assessments for similarity search. Per-user only."""
    client = _get_client()
    col = client.get_or_create_collection(
        "history", embedding_function=_get_embed_fn()
    )
    # clear this user's old entries, then re-index fresh
    try:
        col.delete(where={"user_id": user_id})
    except Exception:
        pass
    ids, docs, metas = [], [], []
    for a in assessments:
        intake = a.get("intake") or {}
        summary = (a.get("report") or "")[:1500]
        doc = (
            f"Date: {a.get('created_at', '')}\n"
            f"Symptoms: {intake.get('SYMPTOMS', '')}\n"
            f"Severity: {intake.get('SEVERITY_1_10', '')}/10\n"
            f"Summary: {summary}"
        )
        ids.append(f"u{user_id}_a{a['id']}")
        docs.append(doc)
        metas.append({"user_id": user_id, "assessment_id": a["id"],
                      "date": str(a.get("created_at", ""))})
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def _format_hits(res: dict) -> list[dict]:
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "id": cid,
            "text": docs[i] if i < len(docs) else "",
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "license": meta.get("license", ""),
        })
    return out


def query_knowledge(query: str, n: int = 5) -> list[dict]:
    """Search the curated technique library. Returns card IDs + text + source lines."""
    col = _get_client().get_or_create_collection(
        "knowledge", embedding_function=_get_embed_fn()
    )
    if col.count() == 0:
        build_index()
        col = _get_client().get_collection(
            "knowledge", embedding_function=_get_embed_fn()
        )
    res = col.query(query_texts=[query], n_results=n)
    return _format_hits(res)


def query_history(query: str, user_id: int, n: int = 5) -> list[dict]:
    """Find THIS user's most similar past assessments. Never another user's."""
    col = _get_client().get_or_create_collection(
        "history", embedding_function=_get_embed_fn()
    )
    if col.count() == 0:
        return []  # first session: no history indexed yet
    res = col.query(query_texts=[query], n_results=n,
                    where={"user_id": user_id})
    return _format_hits(res)
