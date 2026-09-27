# Build log

Problems that actually came up, in the order they came up, with what the fix
was and what it changed about the result.

## The first measurement was 10% task success, and it was all my bugs

The first end-to-end sweep gave task success between 10% and 19% across every
strategy and catalog size. That is a useless measurement: at that floor the
strategies cannot be told apart, and the curves carry no information. I nearly
started tuning the simulated model's noise to raise it, which would have been
fitting the harness to the number I wanted. Looking at individual failures
instead found four straightforward bugs, none of them about model behaviour.

**JSON Schema erases the distinction the argument matcher relied on.** The
simulated model placed a query literal onto a parameter only if their types were
compatible, and it read the type off the rendered schema. Every string parameter
in JSON Schema is `"type": "string"`, so an id parameter, a filesystem path and a
free-text body all look identical, and a literal the extractor had tagged as an
`id` was type-incompatible with all of them. The model therefore invented values
for parameters whose values were sitting in the query. The fix is that a string
parameter accepts any string-representable literal, and placement is decided by
name and description instead. In hindsight this is the correct model of the
problem rather than a workaround: an agent facing real JSON Schema has exactly
this ambiguity, and description text is exactly what resolves it. It is also why
the compression measurement has something to measure at all.

**The URL pattern truncated at the first dot.** `https?://\S+?(?=[\s.,]|$)` is
non-greedy with a lookahead that accepts a dot, so
`https://cb.northwind.example/ingest` matched as `https://cb`. Every webhook task
failed on a wrong argument value that was a prefix of the right one.

**Enum parameters never matched.** The view parser set `type_hint` from
`format` or `type`, both of which are `string` for an enum, so the enum branch of
the matcher was dead code and `format='json'` was never placed.

**The stopword filter was deleting the words that carry the answer.** Context
for a literal was run through the same tokenizer as documents, which drops
`in`, `to`, `at`, `as` and `for`. Those are precisely the words that say which
parameter a value belongs to: "in `#release-train`", "as `csv`", "owned by
`alice@...`". With them gone, placement among several same-typed values was close
to a coin flip. Context now keeps stopwords, and is kept separate from the
document tokenizer rather than sharing it.

Also, nested matches leaked: `_PATHY` matched `example/ingest` inside a URL and
`io/northwind/checkout` inside an image tag, and the standalone enum scanner
matched `urgent` inside the quoted string `'priority=urgent'`. Extraction now
records spans and drops any match strictly contained in a longer one.

After all four, rerank at a 50-tool catalog went from 0.554 to... the same 0.554,
which was a useful shock: the fixes moved the small-catalog numbers a lot
(size 10 went 0.192 to 0.671) but the large-catalog ones much less, because at
1236 tools the failures are dominated by picking the wrong tool rather than by
filling arguments. That is the shape of the final result and I did not expect it
before seeing it.

## Nested catalogs, then per-query catalogs

The first design sampled a size-N catalog as a prefix of one shuffled list, so
size 50 was a subset of size 100. That is nested, which is what makes the curves
comparable point to point, but it means a query whose gold tool sits at position
900 is unanswerable at every size below 900, and the scorable task set changes
with size. Comparing 0.55 at size 50 against 0.33 at size 1236 when the two were
scored on different tasks is not a comparison.

Forcing every gold tool into every subset does not work either: 240 tasks need up
to 240 pinned tools, so at size 10 the "catalog" would be nothing but gold tools.

What works is a per-query catalog. Each task gets its own deterministic ordering
of the whole catalog with the tools it needs pinned to the front, and size N is
the first N of that ordering. Every size is a superset of the one below it, every
task is scorable at every size, and size 10 means what it should mean: the right
tool plus nine distractors. `CatalogWindows` is that.

This made the indexes the bottleneck, since a per-query catalog implies a
per-query index. Re-encoding 1236 documents for each of 240 queries at each of 8
sizes is about ten minutes of embedding per strategy. The indexes are now built
once over the whole catalog and masked to the subset at scoring time. That is
also how a gateway really works, but it does mean corpus statistics including idf
come from the full catalog at every size. That is a confound, so it is stated in
the module docstring rather than left implicit.

## The token counter had to be fitted, not guessed

Token counts are the whole compression measurement, so an estimator wrong in a
format-dependent way would invent the answer. My first structural guess
(alphabetic runs split every four characters, punctuation one token each) was
45% over on average and, worse, 105% over on minified JSON while only 13% over on
pretty JSON. It would have reported minified JSON as *more* expensive than the
original.

A five-feature least-squares fit on 224 recorded Vertex `count_tokens` results
got mean absolute error to 8.4%, but with a negative coefficient on alphabetic
characters, which is collinear nonsense that would not generalise. Going back to
a structural model and fitting its six constants by grid search gave 3.6%. The
committed fixture has a held-out split, which scores 3.8%, so it is not overfit.
Minified JSON is still the worst format at 8.3% over, which understates that
format's measured saving rather than inflating it.

The fixture stores recipes, a format and a list of tool names, not saved strings.
Text is regenerated from the catalog, so `atg calibrate` checks that the
estimator still agrees with the real tokenizer on renderings this repo produces
today, and it fails loudly if the catalog fingerprint has moved.

## Two tests assert limitations

`test_local_embedder_cannot_bridge_a_purely_semantic_gap` was originally written
the other way round and failed. "Give the customer all their money back" retrieves
nothing from the refund family, because the local embedder is a hashed tf-idf
projection over words and character 4-grams and there is no substring shared
between that phrasing and any refund tool. Character n-grams buy morphology, not
meaning. The test now asserts the failure, so the embedding column in
MEASUREMENTS.md is not read as an upper bound on dense retrieval.

The second one was harder to accept. Dropping per-parameter descriptions ought to
cost accuracy, since placing an argument is what those descriptions are for, and
the harness is built so that a rendering which drops them really does deprive the
model. It does not cost anything measurable: the delta is under two points and
changes sign between catalog sizes, +0.038 at 25 tools and -0.004 at 1236. The
reason is structural, not a tuning accident. The simulator places arguments by
overlapping a parameter's words against the words introducing a literal, and this
catalog's generated descriptions are largely restatements of the parameter name
plus the resource noun, so they contribute about as many spurious matches as real
ones. A deterministic overlap score has no way to weigh prose as evidence. A
simulator can show that compression removes information; it cannot price it. That
question is answered by the real-model run and the test says so.

## Recall and task success disagree in one direction only

I built the alias mechanism expecting disagreement in both directions: a second
server exposing the same operation under a different name should sometimes let
the agent finish a task whose gold tool was never retrieved, making recall@k
pessimistic. It never happened. Not once in 1200 attempts.

The reason is obvious in retrospect. A tool that does the same job as another
reads almost the same to a retriever, so it is retrieved *alongside* the tool it
substitutes for, not instead of it. Substitution does happen, 11 times, but every
one of those had the gold tool in the shortlist too, so recall already counted a
hit. The asymmetry is the better result: recall@k is an optimistic upper bound on
task success here, not a proxy for it. Reporting the symmetric claim I set out to
make would have been wrong.

## The proxy calling convention was broken and its first numbers were wrong

Compression formats cannot be expressed as native function declarations, so the
compression comparison uses a single `call_tool(name, arguments)` function with
the rendered candidates in the prompt text. The first run of that scored 0.175
against 0.667 for the same tasks under native declarations, which I nearly wrote
up as a property of the protocol.

It was a prompt bug. The model returned no function call at all on 30 of 40
calls. Told to "choose exactly one tool and call it", and handed one function
that is not any of the tools it was asked to choose between, it reasonably
answered in prose. A convention-specific system prompt that names the indirection,
plus renaming the argument to `arguments_json` and showing an example, took the
same measurement to 0.417. The bad fixture is deleted rather than kept, and the
proxy numbers are only compared against each other.

## Vertex quota

Quota on this project was already under pressure, and the headline run stalled
for about twenty minutes inside the backoff chain at call 31 of 180 before
recovering and finishing all 180 with 3 retries. Two things made that survivable:
the fixture is written every ten calls, so a run killed by quota leaves usable
recorded data rather than nothing, and the runner catches the exception, writes
what it has and reports how far it got. `count_tokens` was never rate limited,
only `generate_content`.

Every real-model number in MEASUREMENTS.md replays offline from a committed
fixture, and a test asserts the replay reproduces the recorded figures exactly.

## Smaller things

The gateway's JSON-RPC parser accepted positional params: `obj.get("params") or
{}` turns an empty array into an empty object before the type check runs, so only
a *non-empty* array was rejected. MCP never uses positional params, so rejecting
them is right; doing it correctly needed checking the value as given.

The replay decider dropped records where the model produced no function call,
because `load_cache` filtered on a truthy name. That turned a recorded prose
answer into a replay miss, which is a different result and would have flattered
the model on exactly the calls it did worst on.

Randomly assigning narrow preconditions to generated tools produced incoherent
pairings, the best being a tool for deleting an account "only valid when sending
to more than 50 recipients in one call". Preconditions now declare which domains
and verbs they apply to. They also carry two phrasings, one for the schema and a
differently-worded one for the query, because a query that quoted the schema
sentence back would hand lexical retrieval the answer for free.

The gateway demo drained the warehouse rate-limit bucket in the section
demonstrating rate limits and then failed three sections later on an unrelated
call. The demo refills the bucket afterwards. Worth noting that the bug was in the
demo, not the limiter.

## What I would do next

The measurement says wrong-tool selection, not retrieval recall, is what breaks
at scale: at 1236 tools the best strategy retrieves the right tool 84% of the
time and completes the task 34% of the time under the simulator, 62% under the
real model. Everything interesting is in that gap, and none of the five
strategies addresses it, because they all stop at handing over a shortlist.
The obvious next thing is disambiguation as its own step, where a second pass
sees only the near-duplicates and the constraint that separates them. The
failure taxonomy already isolates the cases it would have to fix.
