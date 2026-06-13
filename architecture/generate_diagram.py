#!/usr/bin/env python3
"""
generate_diagram.py  —  ContractLens MS3 Architecture
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
    nodesep="0.45",
    ranksep="0.65",
    fontname="Helvetica",
    fontsize="10",
    bgcolor="white",
    pad="0.4",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C_LLM      = ("#FFDCA8", "#C06000")   # orange — LLM agents
C_PYTHON   = ("#C8DCFF", "#2255CC")   # blue — pure-python steps
C_RAG      = ("#B8F0C8", "#196830")   # green — RAG indexes
C_DATA     = ("#ECECEC", "#666666")   # grey — data stores
C_OUT      = ("#FFD0DC", "#AA0033")   # pink — outputs / RunTrace
C_IN       = ("#FFFFFF", "#999999")   # white — inputs
C_TUI      = ("#E8DAFF", "#6E40C0")   # purple — TUI / approval
C_AUDIT    = ("#FFF5C0", "#8A6D00")   # yellow — cross-cutting audit


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

def tui(d, n, lbl):
    d.node(n, lbl, shape="box", style="filled,rounded",
           fillcolor=C_TUI[0], color=C_TUI[1],
           fontname="Helvetica Bold", fontsize="9", margin="0.15,0.1")

def audit(d, n, lbl):
    d.node(n, lbl, shape="hexagon", style="filled",
           fillcolor=C_AUDIT[0], color=C_AUDIT[1],
           fontname="Helvetica", fontsize="9", margin="0.15,0.1")

def e(d, a, b, lbl="", **kw):
    kw.setdefault("color", "#555555")
    kw.setdefault("fontsize", "8")
    kw.setdefault("fontname", "Helvetica")
    kw.setdefault("fontcolor", "#333333")
    d.edge(a, b, lbl, **kw)

def de(d, a, b, lbl=""):    # dashed — retrieval
    e(d, a, b, lbl, style="dashed", color=C_RAG[1], fontcolor=C_RAG[1])

def dot_e(d, a, b, lbl=""): # dotted — build-time / data feed
    e(d, a, b, lbl, style="dotted", color="#999999", fontcolor="#999999")

def tool_e(d, a, b, lbl=""): # tool call edge (orange)
    e(d, a, b, lbl, style="dashed", color=C_LLM[1], fontcolor=C_LLM[1])

def audit_e(d, a, b, lbl=""): # audit edge (yellow)
    e(d, a, b, lbl, style="dotted", color=C_AUDIT[1], fontcolor=C_AUDIT[1])

# ── NODES ─────────────────────────────────────────────────────────────────────

# Inputs
inp(dot, "input",
    "user prompt · contract\nslash commands")

# TUI layer
tui(dot, "tui",
    "TUI / REPL\nuser ⇄ developer mode\n/vector-rag ⇄ /graph-rag\nslash commands · approval prompts\nstreaming output")

# Orchestrator
llm(dot, "orch",
    "Orchestrator Agent\nqwen/qwen3.5-27b (OpenRouter)\nintent routing · tool calls · SSE streamed\ndefault: converse")

# Approval gate
tui(dot, "gate",
    "Approval Gate\n(heavy tools only)\n⚠ run_full_analysis")

# Tools cluster (orchestrator surface)
py(dot, "t_ret",  "tool: retrieve(query, mode,\n      top_k, hyp_id?, label?)")
py(dot, "t_look", "tool: lookup_hypothesis(h_id)")
py(dot, "t_full", "tool: run_full_analysis(contract)\n[requires approval]")

# RAG indexes
rag(dot, "vrag",
    "Vector RAG\nFAISS IndexFlatIP\nall-MiniLM-L6-v2\n32,359 spans")
rag(dot, "grag",
    "GraphRAG\nNetworkX DiGraph\n~19 ConceptNodes\nCITED_FOR (gold)")

# run_full_analysis internal pipeline
py(dot,  "disp",
    "Dispatcher (py)\nbuild 17 HypothesisTasks\n+ pre-fetch RAG per task")
llm(dot, "pool",
    "Hypothesis Agents × 17\nqwen/qwen3.5-9b (OpenRouter)\nN_PARALLEL_AGENTS=5 concurrent\ntools: retrieve · get_span\n→ label · evidence · quote\n   confidence · groundedness")
py(dot,  "agg",
    "Aggregator (py)\nvalidate · apply playbook\ncompute metrics\nflag-only feedback\n(no auto-retry)")

# Playbook
data(dot, "pb", "playbook.yaml\n(MS1 · unedited)")

# Data
data(dot, "train",  "Training Corpus\ntrain.json · 423 NDAs\n32,359 spans\n(index source only)")
data(dot, "tested", "Analyzed Contract\ntest.json · 123 NDAs\n(evidence source)")

# Cross-cutting audit
audit(dot, "rec",
      "Tool-Call Recorder\nname · args · output · count\n(cross-cutting)")

# Outputs
out(dot, "rt_contract",
    "Contract RunTrace\nschema v3.0-ms3\nrunatrace_doc_NNN.json\n17 traces + tool_calls\n+ playbook + metrics")
out(dot, "rt_session",
    "Session RunTrace\nschema v3.0-ms3\nsession_<id>.json\nhistory + tool_calls\n+ approval_events")
out(dot, "ans",
    "Conversational Answer\n[contract] vs [external]\ncitations · streamed")

# ── EXPLICIT RANKS ────────────────────────────────────────────────────────────
with dot.subgraph() as s:
    s.attr(rank="source"); s.node("input")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("tui")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("orch")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("t_ret"); s.node("t_look"); s.node("t_full")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("gate")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("vrag"); s.node("grag"); s.node("disp")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("pool")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("agg"); s.node("pb")

with dot.subgraph() as s:
    s.attr(rank="same"); s.node("rt_contract"); s.node("rt_session"); s.node("ans")

with dot.subgraph() as s:
    s.attr(rank="sink"); s.node("train"); s.node("tested")

# ── EDGES ─────────────────────────────────────────────────────────────────────

# User flow
e(dot, "input", "tui",  "type prompt / slash")
e(dot, "tui",   "orch", "forward prompt + history")

# Orchestrator decides — three tools (+ implicit converse)
tool_e(dot, "orch", "t_ret",  "if retrieval helps")
tool_e(dot, "orch", "t_look", "if analysis cached")
tool_e(dot, "orch", "t_full", "if user intent = full review")

# Approval gate sits in front of heavy tools
e(dot, "t_full", "gate", "⚠ ask user")
e(dot, "gate",   "disp", "approved", color=C_TUI[1], fontcolor=C_TUI[1])

# retrieve → RAG branches
de(dot, "t_ret", "vrag", "mode=vector")
de(dot, "t_ret", "grag", "mode=graph")
de(dot, "vrag",  "t_ret")
de(dot, "grag",  "t_ret")

# lookup feeds from prior analysis output
e(dot, "t_look", "rt_contract", "read cached HypothesisTrace", style="dotted")

# run_full_analysis pipeline
e(dot, "disp", "pool", "17 tasks + RAG context")
e(dot, "pool", "agg",  "17 HypothesisTraces")
e(dot, "pb",   "agg",  "rules", style="dotted", color="#666666")
e(dot, "agg",  "rt_contract")

# Aggregator feedback: failures are FLAGGED on traces, not re-run
e(dot, "agg", "rt_contract",
  "flag failed traces",
  style="dashed", color="#AA3333", fontcolor="#AA3333")

# Dispatcher pre-fetches per-hypothesis RAG
de(dot, "disp", "vrag", "retrieve(Hn, label)")
de(dot, "disp", "grag", "retrieve(Hn, label)")
de(dot, "vrag", "disp")
de(dot, "grag", "disp")

# Hypothesis agents may tool-call retrieve + get_span
de(dot, "pool", "vrag", "retrieve (rare)")
de(dot, "pool", "grag", "retrieve (rare)")
e(dot,  "pool", "tested", "get_span(idx)", style="dotted", color="#666666")

# Orchestrator answers conversationally (implicit, no tool)
e(dot, "orch", "ans", "answer (default)", color=C_LLM[1], fontcolor=C_LLM[1])

# Contracts data feeds
dot_e(dot, "train",  "vrag", "indexed (build-time)")
dot_e(dot, "train",  "grag", "indexed (build-time)")
dot_e(dot, "tested", "orch", "contract text")
dot_e(dot, "tested", "disp", "chunks")

# Tool-call recorder hooks into every agent
audit_e(dot, "orch",  "rec")
audit_e(dot, "pool",  "rec")
audit_e(dot, "disp",  "rec")
audit_e(dot, "agg",   "rec")

# Recorder serialises into both RunTrace flavours
audit_e(dot, "rec", "rt_contract", "tool_calls")
audit_e(dot, "rec", "rt_session",  "tool_calls")

# TUI also writes the session RunTrace (history + approvals)
audit_e(dot, "tui", "rt_session", "history + approval_events")

# ── LEGEND ────────────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_legend") as leg:
    leg.attr(label="Legend", fontname="Helvetica Bold", fontsize="9",
             style="filled", fillcolor="#F8F8F8", color="#BBBBBB", margin="8")

    leg.node("l1", "LLM Agent",        shape="box", style="filled,rounded",
             fillcolor=C_LLM[0], color=C_LLM[1], fontname="Helvetica Bold", fontsize="8")
    leg.node("l2", "Python Step",      shape="box", style="filled",
             fillcolor=C_PYTHON[0], color=C_PYTHON[1], fontname="Helvetica", fontsize="8")
    leg.node("l3", "RAG Index",        shape="cylinder", style="filled",
             fillcolor=C_RAG[0], color=C_RAG[1], fontname="Helvetica", fontsize="8")
    leg.node("l4", "Data Store",       shape="box", style="filled",
             fillcolor=C_DATA[0], color=C_DATA[1], fontname="Helvetica", fontsize="8",
             peripheries="2")
    leg.node("l5", "TUI / Approval",   shape="box", style="filled,rounded",
             fillcolor=C_TUI[0], color=C_TUI[1], fontname="Helvetica Bold", fontsize="8")
    leg.node("l6", "Audit (recorder)", shape="hexagon", style="filled",
             fillcolor=C_AUDIT[0], color=C_AUDIT[1], fontname="Helvetica", fontsize="8")

    leg.node("la", "─── data flow",          shape="plaintext", fontsize="8", fontcolor="#555555")
    leg.node("lb", "- - - retrieval",        shape="plaintext", fontsize="8", fontcolor=C_RAG[1])
    leg.node("lc", "- - - tool call",        shape="plaintext", fontsize="8", fontcolor=C_LLM[1])
    leg.node("ld", "··· build / data feed",  shape="plaintext", fontsize="8", fontcolor="#999999")
    leg.node("le", "··· tool-call audit",    shape="plaintext", fontsize="8", fontcolor=C_AUDIT[1])

    leg.edge("l1", "l2", style="invis")
    leg.edge("l2", "l3", style="invis")
    leg.edge("l3", "l4", style="invis")
    leg.edge("l4", "l5", style="invis")
    leg.edge("l5", "l6", style="invis")
    leg.edge("la", "lb", style="invis")
    leg.edge("lb", "lc", style="invis")
    leg.edge("lc", "ld", style="invis")
    leg.edge("ld", "le", style="invis")

# ── RENDER ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = dot.render(cleanup=True)
    print(f"Written: {path}")
