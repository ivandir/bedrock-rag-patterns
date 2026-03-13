# RAG Patterns Guide

This guide explains each pattern in this library: what problem it solves,
how it works, when to use it, and how to tune it.

---

## Pattern 1: Hybrid Search with Reciprocal Rank Fusion

**Module:** `retrieval.py` — `HybridRetriever`

### The Problem

Pure vector (semantic) search encodes the *meaning* of text into a
high-dimensional embedding, then finds chunks with similar meaning.  This
works well for paraphrase recall — a user asking "how much does the plan
cost?" will surface documents about pricing even if they use the word
"fee" instead of "cost."

But vector search fails silently on exact identifiers:

- A user searching for `SKU-4892-ENT` gets results about enterprise pricing
  in general, because the embedding model doesn't know that `SKU-4892-ENT`
  is a meaningful atom.
- Error codes, part numbers, version strings, and proper nouns are all
  poorly represented in embedding space.

Keyword search (BM25) handles these cases well: it matches exact terms and
assigns high scores to rare, distinctive tokens.

### The Solution: Reciprocal Rank Fusion

Rather than normalising and adding embedding scores and BM25 scores (which
requires careful per-dataset tuning), RRF combines ranked lists:

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

Where `k = 60` (default) and the sum is over each retriever that ranked
document `d`.  Documents that appear in both result sets receive scores from
both terms — a natural double-boost for documents that are *both* semantically
relevant *and* keyword-relevant.

### When to Use

Use hybrid search whenever your corpus contains:
- Product identifiers, SKUs, model numbers
- Error codes or status codes
- Version numbers, dates
- Proper nouns (names of people, companies, laws, standards)
- Technical jargon that may not appear in the embedding model's training data

If your corpus is purely natural language prose (e.g., FAQ answers), pure
vector search may be sufficient.

### Configuration

```python
retriever = HybridRetriever(
    knowledge_base_id="ABCDEF1234",
    opensearch_endpoint="https://abc123.us-east-1.aoss.amazonaws.com",
    n_results=10,       # results per retriever, before merging
    rrf_k=60,           # increase to 90 to reduce rank sensitivity
)
```

**Tuning `rrf_k`:** The constant `k` controls how much weight is given to
the absolute rank versus the relative rank.  Higher `k` means the top result
and the 5th result are treated more similarly.  60 works well across most
corpora; lower values (30) amplify top-rank differences.

**OpenSearch dependency:** The keyword search leg requires `opensearch-py`
and an OpenSearch Serverless collection in the same region.  If you omit
`opensearch_endpoint`, `HybridRetriever` falls back to pure vector search.

---

## Pattern 2: Claude Re-ranking

**Module:** `reranker.py` — `ClaudeReranker`

### The Problem

Retrieval systems are optimised for recall: given a query, return the N chunks
most likely to contain useful information.  But "topically related" is not
the same as "directly answers the question."

Embedding models encode *topics*, not *answers*.  A chunk about "refund
policy overview" and a chunk about "how to initiate a refund" both have
similar embeddings for the query "how do I get a refund" — but only one
answers it.

A cross-encoder re-ranker solves this by jointly encoding the query and each
candidate chunk together, allowing it to reason about their relationship
rather than comparing independent embeddings.

### The Solution: Claude as a Cross-Encoder

We use Claude to score each chunk on a 1–5 relevance scale:

```
5 — Directly and completely answers the query.
4 — Mostly answers the query; minor gaps.
3 — Partially relevant; contains useful information but incomplete.
2 — Tangentially related; unlikely to help answer the query.
1 — Irrelevant.
```

All chunks are sent in a single prompt (batched), which is faster than N
separate calls.  Claude returns a JSON array with a score and one-sentence
reasoning for each chunk.

Chunks below `min_score` (default: 2) are pruned before generation.  This
reduces noise in the generation context and helps the model focus.

### When to Use

Re-ranking provides the most benefit when:

- Your Knowledge Base contains a large, heterogeneous corpus (many topics)
- Retrieval `n_results` is large (10+) to maximise recall
- You observe hallucinations caused by off-topic context confusing the model
- Answer quality is the primary optimisation target (vs. latency or cost)

Skip re-ranking when:
- Latency is critical (re-ranking adds one Claude API call)
- Your corpus is small and tightly scoped to one topic
- You've already tuned embedding quality for your domain

### Configuration

```python
reranker = ClaudeReranker(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",  # faster/cheaper
    min_score=3,   # more aggressive pruning
)
```

**Model choice:** Claude Haiku is fast and inexpensive for this task and
performs well on re-ranking.  Use Sonnet when answer quality is paramount.

**`min_score` tuning:**
- `min_score=1`: No pruning; all chunks passed to generation.
- `min_score=2` (default): Prunes clearly irrelevant chunks.
- `min_score=3`: Aggressive; only keeps highly relevant chunks.

Set `min_score` based on your precision/recall trade-off.  In narrow domains,
3 works well.  In broad corpora where false negatives are expensive (e.g., a
legal Q&A where a missed clause matters), use 2.

---

## Pattern 3: Citation Grounding

**Module:** `pipeline.py` — `RAGPipeline._generate()`

### The Problem

When a model generates an answer from multiple context chunks, it's not
obvious *which* part of the answer came from which source.  This creates
problems for:

- **Auditability**: Can't verify individual claims without re-reading all sources.
- **User trust**: Users can't judge source credibility if sources aren't linked.
- **Debugging**: When an answer is wrong, you can't identify which chunk contributed the error.

### The Solution: Inline Citation Instructions

The generation prompt instructs Claude to add `[Source N]` citations inline,
immediately after each claim.  The pipeline then parses these references and
resolves them to source URIs:

```python
# In the answer:
"Enterprise customers are eligible for a full refund within 90 days [Source 1].
After 90 days, a pro-rated credit is issued [Source 2]."

# Resolved to:
Citation(source_number=1, source_uri="s3://docs/refund-policy.pdf", chunk_text="...")
Citation(source_number=2, source_uri="s3://docs/billing-terms.pdf", chunk_text="...")
```

The citation extraction uses a regex over the answer text and maps source
numbers back to the chunks passed to the generation prompt.

### When to Use

Always.  Citations add negligible latency (they're in the generation prompt,
not a separate call) and dramatically improve auditability.  The only reason
to disable them would be if you need tightly controlled output format where
`[Source N]` notations would break downstream parsing.

---

## Pattern 4: Bedrock Guardrails Integration

**Module:** `guardrails.py` — `GuardrailsFilter`

### The Problem

Content safety and compliance requirements in production RAG systems include:

- **Denied topics**: The system should not answer questions outside its
  intended scope (e.g., a customer service bot should not give financial advice).
- **PII exposure**: Retrieved documents may contain PII (names, SSNs, emails)
  that should be redacted before it reaches the end user.
- **Harmful content**: The model may generate harmful, hateful, or misleading
  content even from benign source documents.
- **Prompt injection**: A malicious user may embed instructions in their query
  to override the system prompt.

### The Solution: ApplyGuardrail at Input and Output

Bedrock Guardrails is a managed service that applies configurable policies to
text.  We call it twice in the pipeline:

**Step 1 — Input check** (`source="INPUT"`):
- Applied to the raw user query.
- Blocks denied topics and prompt injection attempts.
- Fails fast before spending retrieval/generation budget.

**Step 2 — Output check** (`source="OUTPUT"`):
- Applied to the generated answer.
- Catches PII that leaked from source documents into the answer.
- Catches harmful content in the model's response.

```python
guardrail = GuardrailsFilter(
    guardrail_id="gr-abc123",
    guardrail_version="1",          # use a published version in production
    raise_on_intervention=False,    # handle programmatically vs. raising
)

# Check input
input_result = guardrail.check_input(user_query)
if not input_result.passed:
    return "I can't help with that."

# Generate answer ...

# Check output
output_result = guardrail.check_output(answer)
if not output_result.passed:
    return output_result.output_text or "Response blocked by content filter."
```

### When to Use

Guardrails are mandatory in regulated industries (healthcare, finance, legal)
and any public-facing system.  Even in internal tools, guardrails against PII
exposure are worth the minimal latency overhead (~50ms typical).

**Version management:** Use `"DRAFT"` during development.  Publish a
numbered version for production deployments and pin to it.  This prevents
a guardrail configuration change from silently altering production behaviour.

---

## Pattern 5: Hallucination Detection

**Module:** `hallucination.py` — `HallucinationDetector`

### The Problem

RAG reduces hallucination by providing relevant context, but it does not
eliminate it.  Models can still:

- Synthesise information across chunks in a way no single chunk supports.
- State a specific number, date, or name confidently when the actual value
  is absent from the context.
- Ignore contradictory evidence in the context and give a fluent but wrong answer.

For high-stakes domains (legal, medical, financial), a single unsupported
claim can have serious consequences.

### The Solution: Claude as Fact-Checker

After generation, we ask Claude to break the answer into individual factual
claims and verify each one against the source chunks:

| Status | Meaning |
|---|---|
| `SUPPORTED` | Directly stated in at least one source chunk |
| `INFERRED` | Logically follows from the chunks; not verbatim |
| `UNSUPPORTED` | Not present in and not inferable from the chunks |

The overall risk level is derived from the count of unsupported claims:
- `low`: 0 unsupported claims.
- `medium`: 1–2 unsupported, or >30% inferred.
- `high`: 3+ unsupported, or any unsupported key fact.

### When to Use

Hallucination detection is recommended for high-stakes domains where a
wrong answer has significant consequences.  The cost is one additional
Claude API call per pipeline invocation.

For systems where latency is critical, consider running hallucination
detection asynchronously (log for review) rather than blocking the
response.  You can still surface a `hallucination_risk` label in the UI
based on a fast heuristic (e.g., low confidence score from re-ranking) and
do the detailed check in the background.

### Limitations

The detector itself uses Claude, which means it shares Claude's limitations:
- It may miss subtle logical leaps.
- It may flag correct inferences as unsupported.
- It is not a substitute for human review in truly critical decisions.

Use it as a filter that catches the most egregious fabrications and as a
signal for logging and monitoring, not as a legal or medical compliance tool.
