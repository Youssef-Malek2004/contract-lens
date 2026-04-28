#!/usr/bin/env python3
"""
Index correctness tests for MS2 RAG pipelines.

Run from project root:
    python test_indexes.py
"""
import json
import sys
from pathlib import Path

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"{name}" + (f" — {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        failures.append(msg)


# ---------------------------------------------------------------------------
# 1. normalize_hypothesis_id unit tests
# ---------------------------------------------------------------------------
print("\n── 1. normalize_hypothesis_id ──────────────────────────────────────────")
from src.rag_vector import normalize_hypothesis_id as norm_v
from src.rag_graph  import normalize_hypothesis_id as norm_g
from src.constants  import NDA_TO_H, H_TO_NDA

KNOWN = {
    "H01": "nda-1",  "H05": "nda-5",  "H06": "nda-7",
    "H07": "nda-8",  "H08": "nda-10", "H12": "nda-15",
    "H17": "nda-20",
}
for hid, expected_nda in KNOWN.items():
    for fn, label in [(norm_v, "vector"), (norm_g, "graph")]:
        got = fn(hid)
        check(f"{label}: {hid} → {expected_nda}", got == expected_nda,
              f"got {got!r}")

# zero-padding and format variants
check("vector: H6 → nda-7",    norm_v("H6")    == "nda-7")
check("vector: nda-07 → nda-7", norm_v("nda-07") == "nda-7")
check("vector: nda07 → nda-7",  norm_v("nda07")  == "nda-7")
check("graph:  H6 → nda-7",    norm_g("H6")    == "nda-7")
check("None → None",            norm_v(None) is None and norm_g(None) is None)

# Ensure dead nda keys (gaps) are not produced by any H-id
DEAD_KEYS = {"nda-6", "nda-9", "nda-14"}
for hid in H_TO_NDA:
    got_v = norm_v(hid)
    got_g = norm_g(hid)
    check(f"vector: {hid} not a dead key", got_v not in DEAD_KEYS, f"got {got_v!r}")
    check(f"graph:  {hid} not a dead key", got_g not in DEAD_KEYS, f"got {got_g!r}")


# ---------------------------------------------------------------------------
# 2. Vector index structural checks
# ---------------------------------------------------------------------------
print("\n── 2. Vector index structure ───────────────────────────────────────────")
import faiss, numpy as np

meta_v = json.load(open("data/indexes/vector/metadata.json"))
faiss_idx = faiss.read_index("data/indexes/vector/faiss.index")

spans = []
with open("data/indexes/vector/spans.jsonl") as f:
    for line in f:
        spans.append(json.loads(line))

check("FAISS ntotal == metadata index_size",
      faiss_idx.ntotal == meta_v["index_size"],
      f"faiss={faiss_idx.ntotal}, meta={meta_v['index_size']}")

check("spans.jsonl row count == FAISS ntotal",
      len(spans) == faiss_idx.ntotal,
      f"jsonl={len(spans)}, faiss={faiss_idx.ntotal}")

check("FAISS dimension == 384",
      faiss_idx.d == 384, f"got {faiss_idx.d}")

# Annotation key format — must all be nda-keys, never H-keys or dead nda keys
annotated = [s for s in spans if s["hypothesis_annotations"]]
all_keys = {k for s in annotated for k in s["hypothesis_annotations"]}
bad_h_keys    = {k for k in all_keys if k.upper().startswith("H")}
bad_dead_keys = all_keys & DEAD_KEYS

check("No H-format keys in hypothesis_annotations",
      not bad_h_keys, f"found: {bad_h_keys}")
check("No dead nda keys (nda-6/9/14) in hypothesis_annotations",
      not bad_dead_keys, f"found: {bad_dead_keys}")

# nda-7 annotations exist (H06 — was the primary broken case)
nda7_spans = [s for s in annotated if "nda-7" in s["hypothesis_annotations"]]
check("nda-7 annotations present in spans.jsonl",
      len(nda7_spans) > 0, f"found {len(nda7_spans)} spans")

# All 17 expected nda keys appear at least once
expected_nda_keys = set(NDA_TO_H.keys())
check("All 17 nda keys appear in spans.jsonl",
      expected_nda_keys.issubset(all_keys),
      f"missing: {expected_nda_keys - all_keys}")

# Spot-check: no index misalignment (FAISS result idx lines up with spans[idx])
dummy_vec = np.zeros((1, 384), dtype=np.float32)
dummy_vec[0, 0] = 1.0
_, idxs = faiss_idx.search(dummy_vec, 1)
first_hit = int(idxs[0][0])
check("FAISS index result references a valid spans row",
      0 <= first_hit < len(spans), f"hit idx={first_hit}")


# ---------------------------------------------------------------------------
# 3. Graph index structural checks
# ---------------------------------------------------------------------------
print("\n── 3. Graph index structure ────────────────────────────────────────────")
import pickle

meta_g  = json.load(open("data/indexes/graph/metadata.json"))
hyp_idx = json.load(open("data/indexes/graph/hypothesis_index.json"))
con_idx = json.load(open("data/indexes/graph/concept_index.json"))
G       = pickle.load(open("data/indexes/graph/graph.pkl", "rb"))

check("17 HypothesisNodes in graph",
      meta_g["hypothesis_count"] == 17, f"got {meta_g['hypothesis_count']}")

check("19 ConceptNodes in graph",
      meta_g["concept_count"] == 19, f"got {meta_g['concept_count']}")

# All HypothesisNodes have non-empty text (was broken before fix 3)
hyp_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "HypothesisNode"]
empty_text = [n for n in hyp_nodes if not G.nodes[n].get("text", "")]
check("All HypothesisNodes have non-empty text",
      not empty_text, f"empty: {empty_text}")

# INVOLVES edges exist (were zero before fix 3)
involves_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "INVOLVES"]
check("INVOLVES edges exist (> 0)",
      len(involves_edges) > 0, f"found {len(involves_edges)}")

# CITED_FOR edges use nda-keys, not H-keys and not dead keys
cited_targets = {v for u, v, d in G.edges(data=True) if d.get("edge_type") == "CITED_FOR"}
bad_cited = {t for t in cited_targets if not t.startswith("hyp:nda-")}
check("All CITED_FOR targets are hyp:nda-* nodes",
      not bad_cited, f"bad targets: {bad_cited}")

# hyp:nda-7 node has CITED_FOR edges pointing to it (H06 — the primary broken case)
nda7_cited = [(u, v) for u, v, d in G.edges(data=True)
              if d.get("edge_type") == "CITED_FOR" and v == "hyp:nda-7"]
check("hyp:nda-7 has CITED_FOR edges",
      len(nda7_cited) > 0, f"found {len(nda7_cited)}")

# hypothesis_index contains entries for all 17 nda keys
hi_keys = set(hyp_idx.keys())
check("All 17 nda keys in hypothesis_index",
      expected_nda_keys.issubset(hi_keys),
      f"missing: {expected_nda_keys - hi_keys}")

# No duplicate span IDs within any hypothesis_index entry
dup_found = False
for nda_key, label_map in hyp_idx.items():
    for label, span_list in label_map.items():
        if len(span_list) != len(set(span_list)):
            dup_found = True
            failures.append(f"Duplicates in hypothesis_index[{nda_key}][{label}]")
            print(f"  {FAIL}  Duplicates in hypothesis_index[{nda_key}][{label}]")
            break
if not dup_found:
    print(f"  {PASS}  No duplicate span IDs in hypothesis_index")

# concept_index has entries for all 19 concepts
from src.rag_graph import CONCEPT_SYNONYMS
check("All 19 concepts in concept_index",
      set(CONCEPT_SYNONYMS.keys()) == set(con_idx.keys()),
      f"missing: {set(CONCEPT_SYNONYMS.keys()) - set(con_idx.keys())}")


# ---------------------------------------------------------------------------
# 4. retrieve() end-to-end tests (no model loading — uses pre-built index)
# ---------------------------------------------------------------------------
print("\n── 4. retrieve() end-to-end ────────────────────────────────────────────")
from src.rag_vector import retrieve as vector_retrieve
from src.rag_graph  import retrieve as graph_retrieve

# 4a. Free-text query — basic smoke test
v_results = vector_retrieve("confidential information disclosure", top_k=5)
check("vector: free-text returns 5 results", len(v_results) == 5, f"got {len(v_results)}")
check("vector: results have required fields",
      all({"text","doc_id","span_idx","score","hypothesis_annotations"}.issubset(r.keys())
          for r in v_results))

g_results = graph_retrieve("confidential information disclosure", top_k=5)
check("graph: free-text returns results", len(g_results) > 0, f"got {len(g_results)}")

# 4b. hypothesis_id="H06" — was completely broken before fix
v_h06 = vector_retrieve("third party sharing", top_k=5, hypothesis_id="H06", label_filter="ENTAILED")
check("vector: H06 returns results", len(v_h06) > 0, f"got {len(v_h06)}")

# Re-ranking invariant: any result that carries nda-7:ENTAILED must score
# >= any result that doesn't (the bonus only adds, never subtracts).
if v_h06:
    annotated_scores   = [r["score"] for r in v_h06 if r["hypothesis_annotations"].get("nda-7") == "ENTAILED"]
    unannotated_scores = [r["score"] for r in v_h06 if r["hypothesis_annotations"].get("nda-7") != "ENTAILED"]
    rerank_ok = (not annotated_scores or not unannotated_scores or
                 min(annotated_scores) >= max(unannotated_scores) - 0.001)
    ann_min  = f"{min(annotated_scores):.4f}"   if annotated_scores   else "n/a"
    unann_max= f"{max(unannotated_scores):.4f}" if unannotated_scores else "n/a"
    check("vector: H06 annotated results score >= unannotated results",
          rerank_ok,
          f"annotated_min={ann_min}  unannotated_max={unann_max}")

# No result should carry a nda-6 annotation (the old bug's artifact)
bad_nda6 = [r for r in v_h06 if "nda-6" in r["hypothesis_annotations"]]
check("vector: H06 results contain no nda-6 annotations (old bug artifact)",
      not bad_nda6, f"found {len(bad_nda6)} results with nda-6")

g_h06 = graph_retrieve("third party sharing", top_k=5, hypothesis_id="H06", label_filter="ENTAILED")
check("graph: H06 returns results", len(g_h06) > 0, f"got {len(g_h06)}")

# 4c. Re-ranked score must be >= cosine-only score (bonus only adds)
v_plain = vector_retrieve("third party sharing", top_k=5)
v_rerank = vector_retrieve("third party sharing", top_k=5, hypothesis_id="H06", label_filter="ENTAILED")
if v_rerank:
    check("vector: re-ranked top score >= plain top score",
          v_rerank[0]["score"] >= v_plain[0]["score"] - 0.001,
          f"reranked={v_rerank[0]['score']:.4f}, plain={v_plain[0]['score']:.4f}")

# 4d. Non-zero-padded H6 must work identically to H06
v_h6  = vector_retrieve("third party", top_k=3, hypothesis_id="H6")
v_h06b= vector_retrieve("third party", top_k=3, hypothesis_id="H06")
check("vector: H6 and H06 return identical results",
      [r["span_idx"] for r in v_h6] == [r["span_idx"] for r in v_h06b],
      f"H6={[r['span_idx'] for r in v_h6]}  H06={[r['span_idx'] for r in v_h06b]}")

# 4e. graph free-text with no concept match returns []
no_match = graph_retrieve("xyz123 foobar nonsense", top_k=5)
check("graph: unmatched query returns []", no_match == [], f"got {no_match}")


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------
print()
if failures:
    print(f"{'─'*70}")
    print(f"  {len(failures)} test(s) FAILED:")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)
else:
    total = sum(1 for line in open(__file__) if "check(" in line) - 3
    print(f"  All checks passed ✓")
    sys.exit(0)
