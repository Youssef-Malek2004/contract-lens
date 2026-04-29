#!/usr/bin/env python3
"""
generate_diagram.py  —  ContractLens MS2 Architecture
Usage:
    cd contract-lens
    python architecture/generate_diagram.py
Requires:
    conda install -c conda-forge graphviz python-graphviz
"""
from graphviz import Digraph

# ── Canvas ────────────────────────────────────────────────────────────────────
dot = Digraph(name="ContractLens", filename="architecture/architecture", format="pdf")
dot.attr(
    rankdir="TB",
    splines="spline",
    concentrate="false",
    newrank="true",
    nodesep="0.5",
    ranksep="0.7",
    fontname="Helvetica",
    fontsize="10",
    bgcolor="white",
    pad="0.4",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C_LLM      = ("#FFDCA8", "#C06000")   # orange fill, border
C_PYTHON   = ("#C8DCFF", "#2255CC")   # blue fill, border
C_RAG      = ("#B8F0C8", "#196830")   # green fill, border
C_DATA     = ("#ECECEC", "#666666")   # grey fill, border
C_OUT      = ("#FFD0DC", "#AA0033")   # pink fill, border
C_IN       = ("#FFFFFF", "#999999")   # white fill, border

def llm(d, n, lbl):
    d.node(n, lbl, shape="box", style="filled,rounded",
           fillcolor=C_LLM[0], color=C_LLM[1],
           fontname="Helvetica Bold", fontsize="9", margin="0.15,0.1")

def py(d, n, lbl):
    d.node(n, lbl, shape="box", style="filled",
           fillcolor=C_PYTHON[0], color=C_PYTHON[1],
           fontname="Helvetica", fontsize="9", margin="0.15,0.1")

def rag(d, n, lbl):
    d.node(n, lbl, shape="cylinder", style="filled",
           fillcolor=C_RAG[0], color=C_RAG[1],
           fontname="Helvetica", fontsize="9", margin="0.15,0.1")

def data(d, n, lbl):
    d.node(n, lbl, shape="box", style="filled",
           fillcolor=C_DATA[0], color=C_DATA[1],
           fontname="Helvetica", fontsize="8",
           margin="0.1,0.08", peripheries="2")

def inp(d, n, lbl):
    d.node(n, lbl, shape="parallelogram", style="filled",
           fillcolor=C_IN[0], color=C_IN[1],
           fontname="Helvetica Oblique", fontsize="9", margin="0.15,0.1")

def out(d, n, lbl):
    d.node(n, lbl, shape="note", style="filled",
           fillcolor=C_OUT[0], color=C_OUT[1],
           fontname="Helvetica", fontsize="9", margin="0.15,0.1")

def e(d, a, b, lbl="", **kw):
    kw.setdefault("color", "#555555")
    kw.setdefault("fontsize", "8")
    kw.setdefault("fontname", "Helvetica")
    kw.setdefault("fontcolor", "#333333")
    d.edge(a, b, lbl, **kw)

def de(d, a, b, lbl=""):   # dashed — retrieval
    e(d, a, b, lbl, style="dashed", color=C_RAG[1], fontcolor=C_RAG[1])

def dot_e(d, a, b, lbl=""): # dotted — build-time / data feed
    e(d, a, b, lbl, style="dotted", color="#999999", fontcolor="#999999")

# ── NODES ─────────────────────────────────────────────────────────────────────

inp(dot, "input",
    "contract  ·  user prompt\nconversation history  ·  retrieval mode")

# Orchestrator
llm(dot, "orch",
    "Orchestrator\nQwen3-4B\nthinking=ON · tool-calling")

# Two parallel paths
llm(dot, "conv",
    "Conversation Agent\nQwen3-4B\nthinking=ON")

llm(dot, "nli",
    "NLI Core Agent\nQwen3-1.7B + LoRA\nthinking=OFF · adapter ON\n→ 17 labels + evidence spans")

# RAG layer (sits between the two paths and the dispatcher)
rag(dot, "vrag",
    "Vector RAG\nFAISS IndexFlatIP\nall-MiniLM-L6-v2\n32,359 spans")

rag(dot, "grag",
    "GraphRAG\nNetworkX DiGraph\n~19 ConceptNodes\nCITED_FOR (gold)")

# NLI sub-pipeline
py(dot,  "disp",
    "Dispatcher\nbuild_task_queue()\n→ 17 HypothesisTasks\n+ RAG context per task")

llm(dot, "pool",
    "Hypothesis Worker Pool\nQwen3-1.7B\nthinking=ON · adapter OFF\nN_PHYSICAL_AGENTS workers\n→ confidence · quote\n   groundedness · playbook")

py(dot,  "agg",
    "Aggregator\nvalidate schema\ncompute metrics\n→ RunTrace JSON")

# Data stores
data(dot, "train",  "Training Corpus\ntrain.json · 423 NDAs\n32,359 spans\n(index source only)")
data(dot, "tested", "Analyzed Contract\ntest.json · 123 NDAs\n(evidence source)")

# Outputs
out(dot, "runtrace",
    "RunTrace JSON\nschema v2.0-ms2\n17 hypothesis traces\nplaybook · metrics")

out(dot, "convrsp",
    "Conversation Response\n+ updated history")

# ── EXPLICIT RANKS ────────────────────────────────────────────────────────────
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("orch")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("conv"); s.node("nli")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("vrag"); s.node("grag"); s.node("disp")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("pool")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("agg")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("runtrace"); s.node("convrsp")

with dot.subgraph() as s:
    s.attr(rank="source"); s.node("train"); s.node("tested")

# ── EDGES ─────────────────────────────────────────────────────────────────────

# Input → Orchestrator
e(dot, "input", "orch")

# Orchestrator → both paths (tool calls)
e(dot, "orch", "nli",  "run_nli_core")
e(dot, "orch", "conv", "answer_conversationally")

# NLI pipeline
e(dot, "nli",  "disp", "17 labels")
e(dot, "disp", "pool", "task queue ×17")
e(dot, "pool", "agg",  "17 HypothesisTraces")
e(dot, "agg",  "runtrace")

# Conversation output
e(dot, "conv", "convrsp")

# Dispatcher ↔ RAG  (pre-fetch context per hypothesis)
de(dot, "disp", "vrag", "retrieve(Hn, label)")
de(dot, "disp", "grag", "retrieve(Hn, label)")
de(dot, "vrag", "disp")
de(dot, "grag", "disp")

# Conv Agent ↔ RAG
de(dot, "conv", "vrag", "retrieve(prompt)")
de(dot, "conv", "grag", "retrieve(prompt)")
de(dot, "vrag", "conv")
de(dot, "grag", "conv")

# Training corpus → RAG (build time)
dot_e(dot, "train", "vrag", "indexed")
dot_e(dot, "train", "grag", "indexed")

# Analyzed contract → consumers
dot_e(dot, "tested", "nli",  "chunks")
dot_e(dot, "tested", "pool", "chunks")
dot_e(dot, "tested", "conv", "text")

# Orchestrator dispatch_hypothesis_tasks tool call
e(dot, "orch", "disp", "dispatch_hypothesis_tasks", style="dashed", color=C_LLM[1])

# ── LEGEND ────────────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_legend") as leg:
    leg.attr(label="Legend", fontname="Helvetica Bold", fontsize="9",
             style="filled", fillcolor="#F8F8F8", color="#BBBBBB", margin="8")

    leg.node("l1", "LLM Agent", shape="box", style="filled,rounded",
             fillcolor=C_LLM[0], color=C_LLM[1], fontname="Helvetica Bold", fontsize="8")
    leg.node("l2", "Python Fn", shape="box", style="filled",
             fillcolor=C_PYTHON[0], color=C_PYTHON[1], fontname="Helvetica", fontsize="8")
    leg.node("l3", "RAG Index", shape="cylinder", style="filled",
             fillcolor=C_RAG[0], color=C_RAG[1], fontname="Helvetica", fontsize="8")
    leg.node("l4", "Data Store", shape="box", style="filled",
             fillcolor=C_DATA[0], color=C_DATA[1], fontname="Helvetica", fontsize="8",
             peripheries="2")

    leg.node("la", "─── data flow",     shape="plaintext", fontsize="8", fontcolor="#555555")
    leg.node("lb", "- - - retrieval",   shape="plaintext", fontsize="8", fontcolor=C_RAG[1])
    leg.node("lc", "··· build-time",    shape="plaintext", fontsize="8", fontcolor="#999999")

    # Keep legend nodes from affecting main layout
    leg.node("l1"); leg.node("l2"); leg.node("l3"); leg.node("l4")
    leg.edge("l1", "l2", style="invis")
    leg.edge("l2", "l3", style="invis")
    leg.edge("l3", "l4", style="invis")
    leg.edge("la", "lb", style="invis")
    leg.edge("lb", "lc", style="invis")

# ── RENDER ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = dot.render(cleanup=True)
    print(f"Written: {path}")
