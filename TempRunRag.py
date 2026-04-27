from src.rag_vector import retrieve

results = retrieve(
    query="The recipient must keep confidential information secret",
    top_k=5,
)

for i, r in enumerate(results, 1):
    print("=" * 80)
    print("Result:", i)
    print("Doc:", r["doc_id"])
    print("Span:", r["span_idx"])
    print("Score:", r["score"])
    print("Text:", r["text"][:500])
    print("Annotations:", r["hypothesis_annotations"])