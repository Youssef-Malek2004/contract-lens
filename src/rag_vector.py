"""
Member 2 — Vector RAG Pipeline
Embeds training spans with all-MiniLM-L6-v2, indexes with FAISS IndexFlatIP,
and retrieves similar spans for a given query.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.types import RetrievedSpan

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INDEX_DIR = Path("data/indexes/vector")
SPANS_PATH = INDEX_DIR / "spans.jsonl"
FAISS_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded)
# ---------------------------------------------------------------------------
_faiss_index: faiss.IndexFlatIP | None = None
_spans: list[dict] | None = None
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


def _load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load FAISS index and span metadata into module-level singletons."""
    global _faiss_index, _spans

    if _faiss_index is not None and _spans is not None:
        return _faiss_index, _spans

    if not FAISS_PATH.exists() or not SPANS_PATH.exists():
        raise FileNotFoundError(
            f"Vector index not found. Run: python 03_build_index.py --mode vector"
        )

    _faiss_index = faiss.read_index(str(FAISS_PATH))

    _spans = []
    with open(SPANS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            _spans.append(json.loads(line))

    return _faiss_index, _spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_hypothesis_id(hypothesis_id: str | None) -> str | None:
    if hypothesis_id is None:
        return None

    hypothesis_id = hypothesis_id.strip()

    # H07 -> nda-7
    if hypothesis_id.upper().startswith("H"):
        try:
            num = int(hypothesis_id[1:])
            return f"nda-{num}"
        except ValueError:
            return hypothesis_id

    return hypothesis_id


def retrieve(
    query: str,
    top_k: int = 5,
    hypothesis_id: str | None = None,
    label_filter: str | None = None,
) -> list[RetrievedSpan]:
    """
    Retrieve training spans most similar to *query*.

    Steps:
      1. Embed query → L2-normalise → faiss_index.search(vec, top_k * 3)
      2. Load metadata rows from spans.jsonl
      3. If hypothesis_id given: re-rank to prefer spans where
         hypothesis_annotations[hypothesis_id] == label_filter
      4. Return top_k as list[RetrievedSpan]
    """
    index, spans = _load_index()
    model = _get_embed_model()

    # 1. Embed & normalise
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

    # 2. Search (fetch extra candidates for re-ranking)
    fetch_k = min(top_k * 3, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)

    candidates: list[tuple[float, dict]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for missing
            continue
        span_meta = spans[idx]
        candidates.append((float(score), span_meta))

    # 3. Re-rank if hypothesis filter provided
    hypothesis_id = normalize_hypothesis_id(hypothesis_id)
    if hypothesis_id is not None:
        def _rerank_score(item: tuple[float, dict]) -> float:
            cosine_score, meta = item
            annotations = meta.get("hypothesis_annotations", {})
            ann = annotations.get(hypothesis_id)
            bonus = 0.0
            if ann is not None:
                bonus += 0.3  # span is annotated for this hypothesis
                if label_filter and ann == label_filter:
                    bonus += 0.5  # exact label match
            return cosine_score + bonus

        candidates.sort(key=_rerank_score, reverse=True)

    # 4. Build RetrievedSpan list
    results: list[RetrievedSpan] = []
    for score, meta in candidates[:top_k]:
        results.append(
            RetrievedSpan(
                text=meta["text"],
                doc_id=meta["doc_id"],
                span_idx=meta["span_idx"],
                score=round(score, 4),
                hypothesis_annotations=meta.get("hypothesis_annotations", {}),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Index building  (called from 03_build_index.py --mode vector)
# ---------------------------------------------------------------------------

def build_index(train_path: str = "data/train.json") -> None:
    """
    Build the FAISS vector index from training data.

    Produces:
      data/indexes/vector/spans.jsonl
      data/indexes/vector/faiss.index
      data/indexes/vector/metadata.json
    """
    from datetime import datetime, timezone

    print("[vector] Loading training data …")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # ------------------------------------------------------------------
    # Parse spans from train.json
    # The ContractNLI train.json structure:
    #   { "documents": [...], "labels": {...}, "hypotheses": [...] }
    # Each document has "spans" (list of str) and "annotation_sets"
    # ------------------------------------------------------------------
    documents = train_data.get("documents", train_data if isinstance(train_data, list) else [])

    # Handle both top-level list and dict-with-documents formats
    if isinstance(train_data, dict):
        documents = train_data.get("documents", [])
        hypotheses_list = train_data.get("hypotheses", [])
        # Build hypothesis id map: index -> "Hxx"
        hyp_id_map = {}
        for i, h in enumerate(hypotheses_list):
            hyp_id_map[i] = h.get("id", f"H{i+1:02d}")
    else:
        documents = train_data
        hypotheses_list = []
        hyp_id_map = {}

    LABEL_MAP = {
        "Entailment": "ENTAILED",
        "Contradiction": "CONTRADICTED",
        "NotMentioned": "NOT_MENTIONED",
    }

    all_spans: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id", doc.get("doc_id", "unknown"))
        spans_text = doc.get("spans", [])
        annotation_sets = doc.get("annotation_sets", [])

        # Merge all annotation sets for this doc
        merged_annotations: dict[int, dict[str, str]] = {}  # span_idx -> {hyp_id: label}

        for ann_set in annotation_sets:
            annotations = ann_set.get("annotations", {})
            for nda_key, ann_data in annotations.items():
                choice = ann_data.get("choice", "")
                label = LABEL_MAP.get(choice, choice)
                cited_spans = ann_data.get("spans", [])
                for s_idx in cited_spans:
                    if s_idx not in merged_annotations:
                        merged_annotations[s_idx] = {}
                    # Map nda_key to hypothesis id
                    # nda_key is like "nda-1" -> index 0 -> "H01"
                    try:
                        hyp_idx = int(nda_key.split("-")[1]) - 1
                        hyp_id = hyp_id_map.get(hyp_idx, nda_key)
                    except (IndexError, ValueError):
                        hyp_id = nda_key
                    merged_annotations[s_idx][hyp_id] = label

        contract_text = doc.get("text", "")

        for span_idx, span in enumerate(spans_text):
            if isinstance(span, list) and len(span) == 2:
                char_start, char_end = span
                span_text = contract_text[char_start:char_end]
            else:
                char_start, char_end = None, None
                span_text = str(span)

            all_spans.append({
                "text": span_text.strip(),
                "doc_id": str(doc_id),
                "span_idx": span_idx,
                "char_start": char_start,
                "char_end": char_end,
                "hypothesis_annotations": merged_annotations.get(span_idx, {}),
            })

    print(f"[vector] Parsed {len(all_spans)} spans from {len(documents)} documents")

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    model = _get_embed_model()
    texts = [s["text"] for s in all_spans]

    print(f"[vector] Embedding {len(texts)} spans with {EMBEDDING_MODEL_NAME} …")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise so IP == cosine
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    print("[vector] Building FAISS IndexFlatIP …")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # spans.jsonl — row i matches FAISS vector i
    with open(SPANS_PATH, "w", encoding="utf-8") as f:
        for span in all_spans:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    # faiss.index
    faiss.write_index(index, str(FAISS_PATH))

    # metadata.json
    meta = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "index_size": len(all_spans),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[vector] Done — {len(all_spans)} vectors indexed at {INDEX_DIR}")
