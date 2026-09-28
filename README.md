# agent-tool-gateway

Tool retrieval for agents whose catalog is too big for context, scored on whether the
task got done rather than whether the right tool was retrieved. Plus an MCP gateway
that puts the retrieval layer in front of several downstream servers.

An agent with a thousand tools cannot fit their schemas in context, and the failure is
quiet. A model that never sees the tool it needed does not report a missing capability,
it reaches for the nearest thing it can see or invents arguments. So tool selection is
a retrieval problem and it has to be measured against something that notices when the
work does not happen.

## Numbers
```
                        10      100     1236
all tools in context    0.671   0.388   0.246
BM25                    0.662   0.525   0.325
embedding               0.675   0.575   0.338
retrieve then rerank    0.671   0.542   0.338
progressive disclosure  0.671   0.508   0.287
```

All tools in context has recall@8 of 1.000 at every size. The right tool is always
there and task success still collapses, which is the whole point: recall@k is an
optimistic upper bound here, not a proxy.

It goes one way only. Every task that got done had the gold tool in its shortlist, and
about half the attempts that put the right tool in front of the model still failed. No
task was ever solved by a substitute, even though 36 tools are deliberately exposed
twice under different names to make that possible. Substitution happens 11 times, never
on a task that succeeded.

```
full JSON Schema      293,876 tokens          task success 0.400
minified JSON          176,416   39.8% less                0.475
signature + summary     50,589   80.8% less                0.275
signature only          39,432   84.9% less                0.300
```

## Running it

```
uv pip install -e ".[gemini]"
atg bench
python demo/gateway_demo.py
```

The gateway demo runs two downstream servers and kills one, so you can see the breaker
open while the other keeps serving. Works offline on a deterministic model.
