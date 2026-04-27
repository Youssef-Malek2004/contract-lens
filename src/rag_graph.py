"""
Member 3 — GraphRAG Pipeline
Builds a knowledge graph (SpanNode, ConceptNode, HypothesisNode) with
CONTAINS / CITED_FOR / INVOLVES edges, and retrieves spans via
hypothesis-targeted or free-text query paths.
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from src.types import RetrievedSpan

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INDEX_DIR = Path("data/indexes/graph")
GRAPH_PATH = INDEX_DIR / "graph.pkl"
HYPOTHESIS_INDEX_PATH = INDEX_DIR / "hypothesis_index.json"
CONCEPT_INDEX_PATH = INDEX_DIR / "concept_index.json"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# ~19 hardcoded legal concepts with synonym lists
# (source: architecture.yaml — keep in sync)
# ---------------------------------------------------------------------------
CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "confidential_information": [
        "confidential information", "proprietary information", "trade secret",
        "trade secrets", "confidential data", "confidential material",
        "proprietary data", "secret information",
    ],
    "receiving_party": [
        "receiving party", "recipient", "disclosee",
    ],
    "disclosing_party": [
        "disclosing party", "discloser", "provider of information",
    ],
    "third_party": [
        "third party", "third parties", "consultant", "consultants",
        "advisor", "advisors", "agent", "agents", "contractor", "contractors",
        "subcontractor", "subcontractors",
    ],
    "termination": [
        "termination", "expiration", "expiry", "end of agreement",
        "end of the agreement", "terminate", "terminated",
    ],
    "survival": [
        "survival", "survive", "survives", "surviving obligations",
        "post-termination",
    ],
    "return_of_materials": [
        "return of materials", "return or destroy", "return all",
        "destroy all", "return of documents", "destruction of materials",
    ],
    "non_disclosure": [
        "non-disclosure", "nondisclosure", "non disclosure",
        "not disclose", "shall not disclose", "keep confidential",
        "maintain confidentiality",
    ],
    "permitted_use": [
        "permitted use", "permitted purpose", "authorized use",
        "sole purpose", "business purpose", "intended purpose",
    ],
    "injunctive_relief": [
        "injunctive relief", "equitable relief", "specific performance",
        "irreparable harm", "irreparable injury",
    ],
    "indemnification": [
        "indemnification", "indemnify", "hold harmless", "indemnities",
    ],
    "governing_law": [
        "governing law", "governed by", "jurisdiction", "applicable law",
        "laws of the state",
    ],
    "exclusions": [
        "exclusion", "exclusions", "does not include", "shall not apply",
        "publicly available", "public domain", "independently developed",
        "rightfully received",
    ],
    "obligation": [
        "obligation", "obligations", "duty", "duties", "shall",
        "must", "required to", "agrees to",
    ],
    "warranty": [
        "warranty", "warranties", "represents and warrants", "as-is",
        "no warranty",
    ],
    "notice": [
        "notice", "notification", "written notice", "notify", "inform",
    ],
    "amendment": [
        "amendment", "amendments", "modify", "modification", "amend",
        "written amendment",
    ],
    "assignment": [
        "assignment", "assign", "transfer", "delegate", "assignable",
        "non-assignable",
    ],
    "entire_agreement": [
        "entire agreement", "whole agreement", "complete agreement",
        "supersedes", "prior agreements", "prior understandings",
    ],
}

# ---------------------------------------------------------------------------
# 17 hypotheses — pulled from src/constants.py at runtime when available,
# otherwise fallback to nda-1..nda-17 ids.
# ---------------------------------------------------------------------------

def _load_hypotheses() -> dict[str, str]:
    """Return {hypothesis_id: hypothesis_text}."""
    try:
        from src.constants import HYPOTHESES  # type: ignore
        if isinstance(HYPOTHESES, list):
            return {h["id"]: h["text"] for h in HYPOTHESES}
        return HYPOTHESES  # already a dict
    except (ImportError, KeyError):
        # Fallback: will be populated during build from train.json
        return {}


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded)
# ---------------------------------------------------------------------------
_graph: nx.DiGraph | None = None
_hypothesis_index: dict | None = None
_concept_index: dict | None = None


def _load_graph() -> tuple[nx.DiGraph, dict, dict]:
    global _graph, _hypothesis_index, _concept_index

    if _graph is not None and _hypothesis_index is not None and _concept_index is not None:
        return _graph, _hypothesis_index, _concept_index

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "Graph index not found. Run: python 03_build_index.py --mode graph"
        )

    with open(GRAPH_PATH, "rb") as f:
        _graph = pickle.load(f)

    with open(HYPOTHESIS_INDEX_PATH, "r", encoding="utf-8") as f:
        _hypothesis_index = json.load(f)

    with open(CONCEPT_INDEX_PATH, "r", encoding="utf-8") as f:
        _concept_index = json.load(f)

    return _graph, _hypothesis_index, _concept_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_hypothesis_id(hypothesis_id: str | None) -> str | None:
    if hypothesis_id is None:
        return None

    hypothesis_id = hypothesis_id.strip()

    if hypothesis_id.upper().startswith("H"):
        try:
            num = int(hypothesis_id[1:])
            return f"nda-{num}"
        except ValueError:
            return hypothesis_id
        
    if hypothesis_id.lower().startswith("nda") and "-" not in hypothesis_id:
        try:
            num = int(hypothesis_id[3:])
            return f"nda-{num}"
        except ValueError:
            return hypothesis_id

    # nda-07 -> nda-7
    if hypothesis_id.lower().startswith("nda-"):
        try:
            num = int(hypothesis_id.split("-")[1])
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
    Two query paths:

    **Hypothesis-targeted** (hypothesis_id provided):
        hypothesis_index[hypothesis_id][label_filter] → span_ids → load attrs → top_k
        Score: citation frequency across training docs.

    **Free-text** (no hypothesis_id):
        1. Lowercase query → match against synonym lists → matched ConceptNodes
        2. concept_index[canonical_name] → span_ids for each match
        3. Union all candidates
        4. Score: matched_concept_count / total_matched + 0.5 gold bonus
        5. Return top_k
    """
    graph, hyp_index, con_index = _load_graph()

    if hypothesis_id is not None:
        hypothesis_id = normalize_hypothesis_id(hypothesis_id)
        return _retrieve_hypothesis_targeted(
            graph, hyp_index, hypothesis_id, label_filter, top_k
        )
    else:
        return _retrieve_free_text(graph, con_index, query, top_k)


def _retrieve_hypothesis_targeted(
    graph: nx.DiGraph,
    hyp_index: dict,
    hypothesis_id: str,
    label_filter: str | None,
    top_k: int,
) -> list[RetrievedSpan]:
    """Hypothesis-targeted retrieval using pre-computed hypothesis_index."""
    hyp_data = hyp_index.get(hypothesis_id, {})

    if label_filter:
        span_node_ids = hyp_data.get(label_filter, [])
    else:
        # Union all labels for this hypothesis
        span_node_ids = []
        for label_spans in hyp_data.values():
            span_node_ids.extend(label_spans)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for sid in span_node_ids:
            if sid not in seen:
                seen.add(sid)
                deduped.append(sid)
        span_node_ids = deduped

    # Score by citation frequency: count how many CITED_FOR edges this span
    # has for this hypothesis (across potentially multiple annotation sets).
    scored: list[tuple[float, str]] = []
    for nid in span_node_ids:
        if not graph.has_node(nid):
            continue
        # Count CITED_FOR edges from this span to the target hypothesis
        freq = 0
        for _, target, edata in graph.out_edges(nid, data=True):
            if edata.get("edge_type") == "CITED_FOR" and target == f"hyp:{hypothesis_id}":
                freq += 1
        scored.append((freq, nid))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[RetrievedSpan] = []
    for score, nid in scored[:top_k]:
        attrs = graph.nodes[nid]
        results.append(
            RetrievedSpan(
                text=attrs["text"],
                doc_id=attrs["doc_id"],
                span_idx=attrs["span_idx"],
                score=round(float(score), 4),
                hypothesis_annotations=attrs.get("hypothesis_annotations", {}),
            )
        )

    return results


def _retrieve_free_text(
    graph: nx.DiGraph,
    con_index: dict,
    query: str,
    top_k: int,
) -> list[RetrievedSpan]:
    """Free-text retrieval via concept synonym matching."""
    query_lower = query.lower()

    # 1. Match query against synonym lists → matched concept canonical names
    matched_concepts: list[str] = []
    for canonical, synonyms in CONCEPT_SYNONYMS.items():
        for syn in synonyms:
            if syn in query_lower:
                matched_concepts.append(canonical)
                break

    if not matched_concepts:
        # Fallback: try individual words against concept names
        query_words = set(re.findall(r"\w+", query_lower))
        for canonical in CONCEPT_SYNONYMS:
            canonical_words = set(canonical.split("_"))
            if query_words & canonical_words:
                matched_concepts.append(canonical)

    if not matched_concepts:
        return []  # No concepts matched — cannot retrieve via graph

    total_matched = len(matched_concepts)

    # 2. Gather candidate span_node_ids from concept_index
    span_concept_count: dict[str, int] = {}  # span_node_id -> how many concepts matched
    for concept in matched_concepts:
        span_ids = con_index.get(concept, [])
        for sid in span_ids:
            span_concept_count[sid] = span_concept_count.get(sid, 0) + 1

    # 3. Score each candidate
    scored: list[tuple[float, str]] = []
    for nid, concept_hits in span_concept_count.items():
        if not graph.has_node(nid):
            continue
        base_score = concept_hits / total_matched
        # Gold bonus: +0.5 if span has any CITED_FOR edge
        gold_bonus = 0.0
        for _, _, edata in graph.out_edges(nid, data=True):
            if edata.get("edge_type") == "CITED_FOR":
                gold_bonus = 0.5
                break
        scored.append((base_score + gold_bonus, nid))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 4. Return top_k
    results: list[RetrievedSpan] = []
    for score, nid in scored[:top_k]:
        attrs = graph.nodes[nid]
        results.append(
            RetrievedSpan(
                text=attrs["text"],
                doc_id=attrs["doc_id"],
                span_idx=attrs["span_idx"],
                score=round(score, 4),
                hypothesis_annotations=attrs.get("hypothesis_annotations", {}),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Index building  (called from 03_build_index.py --mode graph)
# ---------------------------------------------------------------------------

def build_index(train_path: str = "data/train.json") -> None:
    """
    Build the graph index from training data.

    Produces:
      data/indexes/graph/graph.pkl
      data/indexes/graph/hypothesis_index.json
      data/indexes/graph/concept_index.json
      data/indexes/graph/metadata.json
    """
    print("[graph] Loading training data …")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # ------------------------------------------------------------------
    # Parse structure
    # ------------------------------------------------------------------
    if isinstance(train_data, dict):
        documents = train_data.get("documents", [])
        hypotheses_list = train_data.get("hypotheses", [])
        hyp_id_map: dict[int, str] = {}
        hyp_text_map: dict[str, str] = {}
        for i, h in enumerate(hypotheses_list):
            hid = h.get("id", f"H{i+1:02d}")
            hyp_id_map[i] = hid
            hyp_text_map[hid] = h.get("text", h.get("hypothesis", ""))
    else:
        documents = train_data
        hypotheses_list = []
        hyp_id_map = {}
        hyp_text_map = _load_hypotheses()

    LABEL_MAP = {
        "Entailment": "ENTAILED",
        "Contradiction": "CONTRADICTED",
        "NotMentioned": "NOT_MENTIONED",
    }

    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # 1. Create HypothesisNodes
    # ------------------------------------------------------------------
    all_hypothesis_ids: list[str] = []
    if hyp_text_map:
        for hid, htext in hyp_text_map.items():
            node_id = f"hyp:{hid}"
            G.add_node(node_id, node_type="HypothesisNode", hypothesis_id=hid, text=htext)
            all_hypothesis_ids.append(hid)
    else:
        for i in range(17):
            hid = f"nda{i+1:02d}"
            G.add_node(f"hyp:{hid}", node_type="HypothesisNode", hypothesis_id=hid, text="")
            all_hypothesis_ids.append(hid)

    # ------------------------------------------------------------------
    # 2. Create ConceptNodes + INVOLVES edges (HypothesisNode -> ConceptNode)
    # ------------------------------------------------------------------
    for canonical, synonyms in CONCEPT_SYNONYMS.items():
        node_id = f"concept:{canonical}"
        G.add_node(
            node_id,
            node_type="ConceptNode",
            canonical_name=canonical,
            synonyms=synonyms,
        )

    # INVOLVES: scan each hypothesis text for synonym matches
    for hid in all_hypothesis_ids:
        hyp_node = f"hyp:{hid}"
        hyp_text = G.nodes[hyp_node].get("text", "").lower()
        if not hyp_text:
            continue
        for canonical, synonyms in CONCEPT_SYNONYMS.items():
            for syn in synonyms:
                if syn in hyp_text:
                    G.add_edge(
                        hyp_node,
                        f"concept:{canonical}",
                        edge_type="INVOLVES",
                    )
                    break

    # ------------------------------------------------------------------
    # 3. Create SpanNodes + CONTAINS / CITED_FOR edges
    # ------------------------------------------------------------------
    hypothesis_index: dict[str, dict[str, list[str]]] = {
        hid: {} for hid in all_hypothesis_ids
    }
    concept_index: dict[str, list[str]] = {c: [] for c in CONCEPT_SYNONYMS}

    span_count = 0

    for doc in documents:
        doc_id = str(doc.get("id", doc.get("doc_id", "unknown")))
        spans_text = doc.get("spans", [])
        annotation_sets = doc.get("annotation_sets", [])

        # Merge annotations for this doc
        merged_annotations: dict[int, dict[str, str]] = {}
        for ann_set in annotation_sets:
            annotations = ann_set.get("annotations", {})
            for nda_key, ann_data in annotations.items():
                choice = ann_data.get("choice", "")
                label = LABEL_MAP.get(choice, choice)
                cited_spans = ann_data.get("spans", [])
                try:
                    hyp_idx = int(nda_key.split("-")[1]) - 1
                    hyp_id = hyp_id_map.get(hyp_idx, nda_key)
                except (IndexError, ValueError):
                    hyp_id = nda_key

                for s_idx in cited_spans:
                    if s_idx not in merged_annotations:
                        merged_annotations[s_idx] = {}
                    merged_annotations[s_idx][hyp_id] = label

        contract_text = doc.get("text", "")

        for span_idx, span in enumerate(spans_text):
            span_node_id = f"span:{doc_id}:{span_idx}"

            if isinstance(span, list) and len(span) == 2:
                char_start, char_end = span
                text_stripped = contract_text[char_start:char_end].strip()
            else:
                char_start, char_end = None, None
                text_stripped = str(span).strip()

            text_lower = text_stripped.lower()

            G.add_node(
                span_node_id,
                node_type="SpanNode",
                text=text_stripped,
                doc_id=doc_id,
                span_idx=span_idx,
                char_start=char_start,
                char_end=char_end,
                hypothesis_annotations=merged_annotations.get(span_idx, {}),
            )
            span_count += 1

            # CONTAINS edges: span -> concept (substring match)
            for canonical, synonyms in CONCEPT_SYNONYMS.items():
                for syn in synonyms:
                    if syn in text_lower:
                        G.add_edge(
                            span_node_id,
                            f"concept:{canonical}",
                            edge_type="CONTAINS",
                        )
                        concept_index[canonical].append(span_node_id)
                        break  # one match per concept is enough

            # CITED_FOR edges: span -> hypothesis
            hyp_annotations = merged_annotations.get(span_idx, {})
            for hyp_id, label in hyp_annotations.items():
                hyp_node = f"hyp:{hyp_id}"
                if G.has_node(hyp_node):
                    G.add_edge(
                        span_node_id,
                        hyp_node,
                        edge_type="CITED_FOR",
                        label=label,
                    )
                    # Update hypothesis_index
                    if label not in hypothesis_index.get(hyp_id, {}):
                        hypothesis_index.setdefault(hyp_id, {})[label] = []
                    hypothesis_index[hyp_id][label].append(span_node_id)

    print(f"[graph] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[graph]   SpanNodes: {span_count}")
    print(f"[graph]   ConceptNodes: {len(CONCEPT_SYNONYMS)}")
    print(f"[graph]   HypothesisNodes: {len(all_hypothesis_ids)}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(HYPOTHESIS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(hypothesis_index, f, indent=2)

    with open(CONCEPT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(concept_index, f, indent=2)

    meta = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "span_count": span_count,
        "concept_count": len(CONCEPT_SYNONYMS),
        "hypothesis_count": len(all_hypothesis_ids),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[graph] Done — index written to {INDEX_DIR}")
