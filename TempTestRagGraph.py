from src.rag_graph import retrieve

results = retrieve(
    query="",
    top_k=5,
    hypothesis_id="H7",
    label_filter="CONTRADICTED",
)

print("Number of results:", len(results))

for r in results:
    print(r["doc_id"], r["span_idx"], r["hypothesis_annotations"])