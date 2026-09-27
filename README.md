# agent-tool-gateway

Tool retrieval for agents whose catalog is too large to put in context, measured
on whether the task got done rather than on whether the right tool was retrieved.
Plus an MCP gateway that applies the retrieval layer in front of several
downstream servers.

An agent with a thousand tools cannot fit their schemas in context, and the
failure is quiet. A model that never sees the tool it needed does not report a
missing capability. It reaches for the nearest thing it can see, or fills in
arguments it was never given. So tool selection is a retrieval problem, and
retrieval quality has to be measured against something that notices when the
work does not happen.

## What it measures

A deterministic synthetic catalog of 1236 tools across twenty domains, including
six hand-written clusters of near duplicates where `send_email`,
`send_bulk_email` and `email_draft_create` differ only in a clause of their
description, 36 operations exposed twice under different names by different
servers, and 41 tools that are correct only under a stated condition. Then 240
tasks phrased with synonyms so a query is not a lexical copy of the tool it should
reach, graded end to end: the right tool, called with values actually present in
the request. Five strategies at eight catalog sizes: all tools in context, BM25
implemented here, local deterministic embeddings, retrieve then rerank, and
progressive disclosure where the model first sees one line per tool.

## The numbers

Everything scores the same while the catalog fits in the shortlist. Past that, from
10 to 1236 tools, the all-tools baseline falls from 0.671 task success to 0.246
while reranking holds 0.338, and the baseline's prompt grows from 2,293 tokens to
293,876 while reranking's stays flat around 1,850. Progressive disclosure pays two
model calls and still scores below single-stage retrieval, because its first pass
discards the catalog having seen only one line per tool.

The result worth having is the gap between the two metrics, and its direction. At
the full catalog, 587 of 1200 attempts put the correct tool in front of the model
and still failed, 48.9%. The reverse never happened once: no task was completed
whose gold tool was missing from the shortlist, not even with the same operation
available under a second name on another server. Recall@k is an optimistic upper
bound on task success here, not a proxy for it. The baseline shows that most
bluntly, scoring recall@8 of 1.000 at every catalog size while its task success
falls by two thirds.

The gap is widest where a metric most needs to be honest. Where the correct tool is
correct only under a condition the request states, retrieval succeeds 96% of the
time and the task 18%, because choosing between near-identical tools on one
sentence of description is not a retrieval problem. Where the request omits a
required value and the right move is to ask rather than invent, retrieval succeeds
86% and the task 7%.

Compressing schemas to one-line signatures saves 80.8% of the tokens per query,
84.9% without the summary, taking the whole catalog from 293,876 tokens to 50,589.
The offline token counter is fitted against 240 recorded Vertex `count_tokens`
results and scores 3.8% mean absolute error on a held-out split.

Against real `gemini-2.5-flash-lite` task success runs well above the simulator and
degrades far less, falling 0.050 from 25 to 1236 tools where the simulator falls
0.250. The simulated curves are a lower bound; what transfers is the ordering of
strategies and the metric disagreement, present in both. That run cost $0.104 at
list prices and replays offline from a committed fixture.

MEASUREMENTS.md has all of it, labelled simulated or real with model ids and dates.
NOTES.md is the build log, including the four argument-extraction bugs that had the
first measurement pinned at 10%.

## The gateway

`atg gateway` multiplexes downstream MCP servers behind one endpoint and applies
the retrieval layer, so a client sees four tools rather than the whole estate. Each
server's credential lives in the gateway, not the client. Read-only calls are
cached with a TTL and mutating ones never are, rate limits are per server and tell
the caller when to come back, and a circuit breaker fails fast once a server stops
answering. Tools on a dead server keep appearing in `find_tools` marked
unreachable, which is more use to a caller than pretending they vanished.
`python demo/gateway_demo.py` runs two real downstream subprocesses and kills one
with a real signal.

## Running it

```
uv pip install -e ".[dev]"
pytest
atg bench --report out/report.html
python demo/gateway_demo.py
```

`pytest` passes with no credentials and the real-model tests skip cleanly. The
measurement runs offline on a deterministic simulated model and the recorded Gemini
runs replay from fixtures; re-recording them needs `gcloud auth
application-default login` and `python demo/run_gemini.py`. `atg catalog`,
`atg tasks`, `atg compress`, `atg taxonomy` and `atg calibrate` print the rest.
