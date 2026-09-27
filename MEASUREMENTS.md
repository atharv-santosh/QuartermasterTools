# Measurements

Every number here came from running the code in this repository on this machine.
Nothing is estimated, extrapolated or carried over from anywhere else. Where a
number is a calculation rather than an observation, which is only true of the
dollar figures, it says so.

Two kinds of run appear below and they are never mixed in a table.

**Simulated** runs use `atg.llm_sim`, a deterministic scoring function standing in
for the model. No credentials, no network, byte-identical on re-run.

**Real model** runs use `gemini-2.5-flash-lite` on Vertex AI, project
`research-501307`, location `global`, recorded 2026-09-26. Every one replays
offline from a committed fixture, and
`tests/test_gemini_replay.py::test_the_recorded_run_replays_to_the_numbers_it_recorded`
asserts the replay reproduces the figures below exactly.

Shared setup unless stated otherwise: catalog seed `20260925`, 1236 tools,
fingerprint `265448f30be670de`; 240 tasks, seed `424242`; k=8; full JSON Schema
rendering; local `local-hash-512` embeddings.

---

## 1. Accuracy against catalog size (simulated)

```
atg bench
```

240 tasks per cell, 8 catalog sizes, 5 strategies, 9600 attempts, 114 seconds.

**End-to-end task success**

| strategy | 10 | 25 | 50 | 100 | 200 | 400 | 800 | 1236 |
|---|---|---|---|---|---|---|---|---|
| all tools in context | 0.671 | 0.475 | 0.425 | 0.388 | 0.354 | 0.325 | 0.283 | **0.246** |
| BM25 | 0.662 | 0.583 | 0.550 | 0.525 | 0.475 | 0.438 | 0.367 | 0.325 |
| embedding | 0.675 | 0.625 | 0.596 | 0.575 | 0.504 | 0.463 | 0.400 | **0.338** |
| retrieve then rerank | 0.671 | 0.592 | 0.554 | 0.542 | 0.483 | 0.454 | 0.383 | **0.338** |
| progressive disclosure | 0.671 | 0.588 | 0.525 | 0.508 | 0.429 | 0.388 | 0.321 | 0.287 |

**Recall@8, the same runs**

| strategy | 10 | 25 | 50 | 100 | 200 | 400 | 800 | 1236 |
|---|---|---|---|---|---|---|---|---|
| all tools in context | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| BM25 | 0.971 | 0.963 | 0.954 | 0.925 | 0.912 | 0.883 | 0.842 | 0.812 |
| embedding | 0.971 | 0.954 | 0.938 | 0.917 | 0.900 | 0.871 | 0.842 | 0.792 |
| retrieve then rerank | 0.996 | 0.971 | 0.958 | 0.929 | 0.921 | 0.904 | 0.879 | **0.842** |
| progressive disclosure | 0.975 | 0.904 | 0.825 | 0.767 | 0.713 | 0.642 | 0.571 | 0.533 |

Recall@8 for the all-tools baseline is 1.000 by construction: it retrieves
nothing, so the gold tool is always "in the shortlist". That row is the cleanest
demonstration that recall@k is not a measure of anything useful on its own.

The curves cross between 10 and 25 tools. At 10 tools the catalog fits inside k,
so every strategy shows the model the same set and they score identically, which
is a sanity check on the harness rather than a result
(`test_strategies_agree_when_the_catalog_fits_inside_k`). Past that, the baseline
falls faster than anything else and ends 37% below the best retrieval strategy in
relative terms.

**Mean prompt tokens per query**

| strategy | 10 | 25 | 50 | 100 | 200 | 400 | 800 | 1236 | calls |
|---|---|---|---|---|---|---|---|---|---|
| all tools in context | 2,293 | 5,843 | 11,794 | 23,649 | 47,476 | 95,086 | 190,123 | **293,876** | 1 |
| BM25 | 1,273 | 1,582 | 1,723 | 1,771 | 1,776 | 1,797 | 1,830 | 1,858 | 1 |
| embedding | 1,830 | 1,873 | 1,882 | 1,883 | 1,883 | 1,883 | 1,889 | 1,871 | 1 |
| retrieve then rerank | 1,822 | 1,855 | 1,868 | 1,866 | 1,853 | 1,845 | 1,850 | **1,853** | 1 |
| progressive disclosure | 1,968 | 2,160 | 2,504 | 3,221 | 4,613 | 7,403 | 12,997 | 19,097 | 2 |

The baseline grows 128 times over the sweep while reranking grows 1.02 times.
Progressive disclosure is the interesting middle: its prompt is 10 times smaller
than the baseline's at the full catalog but still 10 times larger than a
shortlist's, because a one-line index still has one line per tool. It also pays
two model calls instead of one and scores *worse* than single-stage retrieval
(0.287 against 0.338), because its first stage has only a one-line summary to
choose from and discards the rest of the catalog before anything has seen a
schema. Its recall@8 at 1236 tools is 0.533 against reranking's 0.842.

---

## 2. Recall@k and task success disagree, in one direction

At 1236 tools, across all five strategies, 1200 attempts:

| | count | share |
|---|---|---|
| recall@8 counted a hit and the task still failed | **587** | 48.9% |
| recall@8 counted a miss and the task still succeeded | **0** | 0.0% |
| task completed by a tool other than the one recall scored against | 11 | 0.9% |

The asymmetry is the result. Recall@k is an **optimistic upper bound** on task
success here, not a proxy for it. Every task the agent completed had the gold tool
in its shortlist, and roughly half of the attempts that put the right tool in
front of the model still did not get the job done.

The second row is zero despite the catalog being built to make it non-zero. 36
tools are exposed twice, once by their own service and once by a legacy bridge
under a different name and different wording, and a task is graded as complete if
either is called. Substitution does happen, 11 times, but every one of those had
the gold tool in the shortlist as well, so recall already counted a hit. A tool
that does the same job as another reads almost the same to a retriever, so it
arrives alongside the tool it substitutes for rather than instead of it.

Why the 587 failed anyway:

| | count |
|---|---|
| wrong tool selected from the shortlist | 331 |
| right tool nowhere near the shortlist (baseline, which retrieves nothing) | 154 |
| near-duplicate confusion | 54 |
| hallucinated argument | 18 |
| missing argument | 11 |
| wrong argument value | 10 |
| right tool just below the cutoff | 9 |

Split by what the task was testing, for retrieve-then-rerank at 1236 tools, the
gap is very unevenly distributed:

| task kind | n | recall@8 | task success | gap |
|---|---|---|---|---|
| conditional, correct only under a stated condition | 28 | 0.964 | **0.179** | 0.785 |
| underspecified, a required value absent from the request | 28 | 0.857 | **0.071** | 0.786 |
| family, a hand-written near-duplicate cluster | 48 | 0.688 | 0.458 | 0.230 |
| aliased, the same operation on two servers | 36 | 0.833 | 0.417 | 0.416 |
| plain | 100 | 0.880 | 0.370 | 0.510 |

Those first two rows are the whole argument. On conditional tasks retrieval
almost never fails (0.964) and the task almost always does (0.179), because the
detail that separates the right tool from its sibling is one sentence in a
description and choosing between them is not a retrieval problem at all. On
underspecified tasks the correct behaviour is to ask for the missing value rather
than invent one, and the agent gets that right twice out of 28.

---

## 3. Schema compression (simulated for tokens, real model for accuracy)

```
atg compress --size 1236
atg compress --size 25
```

Token costs are the same measurement at both sizes, since only the rendering
changes. Same retriever, same k, same tasks.

| rendering | tokens/query | tokens/tool | whole catalog | saving |
|---|---|---|---|---|
| full JSON Schema | 1,853 | 232 | 293,876 | 0.0% |
| minified JSON Schema | 1,116 | 140 | 176,416 | **39.8%** |
| signature plus one-line summary | 355 | 44 | 50,589 | **80.8%** |
| signature only | 280 | 35 | 39,432 | **84.9%** |

A signature is `send_email(to:email!, subject:str!, body:str!) - Send a single
email message immediately.` A trailing `!` marks required; `a|b|c` lists allowed
values. It keeps the tool's purpose and its argument names, types and
requiredness, and drops the per-parameter prose and the JSON punctuation.

Putting the whole 1236-tool catalog in context costs 293,876 tokens as JSON
Schema and 50,589 as signatures. On gemini-2.5-flash-lite list prices that is
2.94 cents against 0.51 cents per query, before any output.

**Accuracy cost, simulated: not resolvable.** Dropping per-parameter descriptions
moves task success by under two points and the sign flips with catalog size:
signature is +0.0375 at 25 tools and -0.0042 at 1236; signature-only is +0.0208
and -0.0208. Minifying the JSON is exactly 0.0000 at both sizes, as it must be,
since whitespace is not information and the parsed view is identical either way.
That zero is a useful check that nothing but the rendering reaches the model.

The simulator cannot price description-dropping, and this is structural rather
than a tuning accident. It places arguments by overlapping a parameter's words
against the words introducing a literal in the query, and this catalog's generated
descriptions largely restate the parameter name plus the resource noun, so they
contribute about as many spurious matches as real ones. A deterministic overlap
score has no way to weigh prose as evidence the way a language model does.
`test_the_simulator_cannot_resolve_the_accuracy_cost_of_dropping_descriptions`
asserts both the small magnitude and the sign flip so this stays recorded.

**Accuracy cost, real model.** See section 5.

---

## 4. Real model against catalog size

```
python demo/run_gemini.py --n 60 --sizes 25,200,1236
python demo/compare_sim_vs_real.py
```

`gemini-2.5-flash-lite`, native per-tool function declarations, temperature 0,
retrieve-then-rerank, k=8, 60 of the 240 tasks (sample seed 99), recorded
2026-09-26. Fixture `tests/fixtures/gemini_headline.json`, 180 calls.

| catalog size | recall@8 | simulated task success | gemini-2.5-flash-lite | gap |
|---|---|---|---|---|
| 25 | 1.000 | 0.533 | **0.667** | +0.133 |
| 200 | 0.900 | 0.417 | **0.667** | +0.250 |
| 1236 | 0.817 | 0.283 | **0.617** | +0.333 |

Recall is identical for both because retrieval does not depend on the model; the
comparison script asserts that rather than assuming it.

**The simulator is pessimistic, and increasingly so.** Over 25 to 1236 tools
recall falls 0.183, simulated task success falls 0.250, and the real model's falls
0.050. The real model barely degrades with catalog size once retrieval has done
its job: almost all of its loss is retrieval loss. The simulator's extra
steepness comes from its `MIDDLE_PENALTY` term, which discounts candidates buried
in the centre of a long candidate list. That term is a modelling assumption
borrowed from published behaviour, not something measured here, and this table is
the evidence that it overstates the effect at k=8.

So the simulated curves in section 1 should be read as a lower bound on what a
real model does, and the ordering of strategies rather than the absolute levels is
what transfers. What does transfer is the disagreement: at 25 tools the real model
has recall 1.000 and task success 0.667, so a third of the attempts had the right
tool in hand and did not finish the job.

Real-model failure distribution at 1236 tools, 60 tasks: 8 deep retrieval miss,
5 wrong tool selected, 4 wrong argument value, 3 right tool just below the cutoff,
1 hallucinated argument, 1 missing argument, 1 no function call returned.
Retrieval misses and wrong-tool selection are 13 of the 23 failures.

**Cost.** 180 calls, 131,875 prompt tokens, 227,458 output tokens (which includes
thinking tokens), 359,333 total, 3 retries against 429 RESOURCE_EXHAUSTED. At
list prices of $0.10 and $0.40 per million tokens that is **$0.104**. That dollar
figure is a calculation from measured token counts, not a billing statement.

---

## 5. Real model, schema compression

```
for f in json_schema compact_json signature oneline; do
  python demo/run_gemini.py --n 40 --sizes 25 --fmt $f --convention proxy
done
```

`gemini-2.5-flash-lite`, 40 tasks, catalog size 25, k=8, recorded 2026-09-26.

This run uses the `proxy` calling convention: one `call_tool(name,
arguments_json)` declaration with the rendered candidates in the prompt text. A
native function declaration cannot represent a one-line signature, so comparing
renderings under native declarations would not be comparing renderings. Proxy
numbers are only comparable against each other, never against section 4's native
numbers, and they are lower across the board because serialising arguments into a
JSON string is a harder protocol than having the SDK do it.

PARTIAL_COMPRESSION_TABLE

---

## 6. Failure taxonomy

```
atg taxonomy --sizes 1236
```

All five strategies at 1236 tools, 832 failures out of 1200 attempts.

| failure mode | count | share |
|---|---|---|
| right tool nowhere near the shortlist | 347 | 41.7% |
| wrong tool selected from the shortlist | 331 | 39.8% |
| right tool just below the cutoff | 61 | 7.3% |
| near-duplicate confusion | 54 | 6.5% |
| hallucinated argument after a miss | 18 | 2.2% |
| missing argument | 11 | 1.3% |
| wrong argument value | 10 | 1.2% |

Definitions, since the distinction matters: **nowhere near the shortlist** means
the gold tool ranked outside 3k or was unranked; **just below the cutoff** means
it ranked between k+1 and 3k, which is the band a larger k would have fixed;
**near-duplicate confusion** means the tool chosen was in the gold tool's
hand-written confusable family; **wrong tool selected** means it was not.

By strategy, the mix changes completely, which is the point of splitting it:

| strategy | failures | top three modes |
|---|---|---|
| all tools in context | 181 | nowhere near 154, just below cutoff 9, wrong tool 6 |
| BM25 | 162 | wrong tool 94, nowhere near 28, just below cutoff 17 |
| embedding | 159 | wrong tool 89, nowhere near 30, just below cutoff 20 |
| retrieve then rerank | 159 | wrong tool 98, nowhere near 23, near duplicate 15 |
| progressive disclosure | 171 | nowhere near 112, wrong tool 44, near duplicate 7 |

The baseline's 154 "nowhere near" failures need reading carefully. It retrieves
nothing, so its gold rank is the tool's position in the catalog as presented,
and a gold tool sitting deep in a 1236-entry list is functionally unreachable
even though it is technically in context. That is the failure mode the project is
about, and it is invisible to recall@k, which scores that same row 1.000.

For retrieval strategies the dominant mode is wrong tool selected from a
shortlist that contained the right one. Adding better retrieval does not fix it.
Raising k fixes only the 61 just-below-cutoff cases.

By task kind, all strategies at 1236 tools:

| task kind | failures | top three modes |
|---|---|---|
| plain | 335 | wrong tool 165, nowhere near 156, just below cutoff 14 |
| family | 131 | nowhere near 46, wrong tool 43, just below cutoff 32 |
| underspecified | 130 | wrong tool 57, nowhere near 37, hallucinated argument 18 |
| conditional | 119 | nowhere near 46, near duplicate 44, wrong tool 17 |
| aliased | 117 | nowhere near 62, wrong tool 49, just below cutoff 4 |

All 18 hallucinated arguments are on underspecified tasks, which is where the
request genuinely omits a required value and inventing one is the failure. All 44
near-duplicate confusions on conditional tasks are the agent choosing a sibling
whose narrow precondition the request did not assert, or missing the one it did.

---

## 7. Token estimator calibration

```
atg calibrate
```

The offline estimator in `atg.tokens` against 240 recorded Vertex `count_tokens`
results on real catalog renderings, `gemini-2.5-flash-lite`, recorded 2026-09-26.
Six constants fitted by grid search on the 168-sample fit split.

| split | n | mean abs error | p95 | max | mean ratio |
|---|---|---|---|---|---|
| fit | 168 | 3.67% | 10.75% | 16.22% | 1.018 |
| **holdout** | **72** | **3.80%** | 11.01% | 15.60% | 1.020 |

| rendering | n | mean abs error | mean ratio |
|---|---|---|---|
| full JSON Schema | 60 | 1.80% | 0.983 |
| minified JSON Schema | 60 | **8.25%** | 1.083 |
| signature plus summary | 60 | 2.05% | 1.000 |
| signature only | 60 | 2.74% | 1.008 |

Holdout error matching fit error means it is not overfit. Minified JSON is the
worst case at 8.3% over, which **understates** that format's saving in section 3
rather than inflating it: correcting for the bias would move its 39.8% saving up,
not down. The fixture stores recipes, not saved strings, so `atg calibrate`
re-renders from the catalog and fails loudly if the catalog fingerprint moves.

---

## 8. Gateway

```
python demo/gateway_demo.py
```

Two downstream MCP servers as real subprocesses over pipes, nine tools between
them. Observed in that run:

* The client's tool surface is **4 tools** (`find_tools`, `describe_tool`,
  `call_tool`, `gateway_health`) against 9 downstream, and it does not grow with
  the estate. With `--no-retrieval` it is 13.
* Response cache: first `get_stock_level` via `upstream`, second via `cache`.
  `create_purchase_order` goes via `upstream` both times, because mutating tools
  are never cached. A cache hit spends no rate-limit token.
* Per-server auth: `mailroom` requires `ATG_MAILROOM_TOKEN` in the gateway
  process, `warehouse` requires nothing. With the token removed the mailroom call
  fails with code -32012 and the warehouse call still succeeds. The token appears
  nowhere in the client's tool surface.
* Rate limiting: with 3 tokens left, 3 calls succeeded and the 4th failed with
  code -32011 and `{"retry_after_s": 0.5}`.
* Degradation: after `SIGKILL` to the mailroom process, 4 calls failed with code
  -32010, the breaker tripped once and went open, and the warehouse kept serving
  throughout. `find_tools` still returned mailroom tools, marked
  `reachable: false` rather than hidden, and its final health showed
  `mailroom: alive false, breaker open, trips 1` beside
  `warehouse: alive true, breaker closed, 1 cache hit`.
* A server that cannot start does not stop the gateway: with the mailroom token
  absent at startup its tools are absent and `find_tools` returns only warehouse
  tools.

The demo wrote a 26-event trajectory conforming to `docs/trajectory-schema.md`.

---

## 9. Reproducing

```
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
pytest                    # 130 passed, 1 skipped, no credentials
atg bench --report out/report.html
atg compress --size 25
atg taxonomy --sizes 1236
atg calibrate
python demo/gateway_demo.py
python demo/compare_sim_vs_real.py
```

Everything above except sections 4 and 5 is produced by those commands with no
credentials and no network. Sections 4 and 5 replay from committed fixtures, also
with no credentials. Re-recording them needs `gcloud auth
application-default login` and `python demo/run_gemini.py`.
