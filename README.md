# bedrock-rag-patterns

Production-ready RAG patterns on AWS Bedrock. Not tutorials — the patterns that survive contact with real data.

## The Problem

Most RAG tutorials show you the happy path: embed some docs, do a vector search, pass chunks to an LLM. Production RAG breaks in ways those tutorials don't cover:

- **Vector search alone misses exact keyword matches** — a user searching for a product SKU or an error code gets poor results because semantic similarity fails on identifiers.
- **Retrieval surfaces related-but-wrong chunks** — a re-ranking step based purely on embedding distance doesn't account for what actually answers the question.
- **Models hallucinate even when context is provided** — the context window is long, relevant chunks may be buried, and the model fills gaps with confident-sounding fiction.
- **Output guardrails are bolted on as an afterthought** — by the time a guardrails violation is caught, you've already spent inference budget on a bad response.

This library gives you composable Python classes for each of these failure modes, wired together in a pipeline you can drop into a real application.

## Patterns

| Pattern | Module | When to Use |
|---|---|---|
| Hybrid Search (RRF) | `retrieval.py` | Any corpus with identifiers, codes, or proper nouns |
| Claude Re-ranking | `reranker.py` | When top-k retrieval quality bottlenecks answer quality |
| Hallucination Detection | `hallucination.py` | High-stakes domains (legal, medical, finance) |
| Bedrock Guardrails | `guardrails.py` | Regulated industries; PII or harmful content filtering |
| Full Pipeline | `pipeline.py` | Production: all patterns composed in the right order |

See [`docs/patterns.md`](docs/patterns.md) for a detailed explanation of each pattern, trade-offs, and tuning guidance.

## Requirements

- Python 3.11+
- AWS credentials with permissions for `bedrock-runtime`, `bedrock-agent-runtime`, and optionally `opensearchserverless`
- An Amazon Bedrock Knowledge Base (for retrieval)
- Optionally: an Amazon Bedrock Guardrail

## Installation

```bash
pip install bedrock-rag-patterns
```

Or from source:

```bash
git clone https://github.com/ivandir/bedrock-rag-patterns
cd bedrock-rag-patterns
pip install -e .
```

## Quick Start

```python
from bedrock_rag import RAGPipeline

pipeline = RAGPipeline(
    knowledge_base_id="ABCDEF1234",
    guardrail_id="gr-abc123",        # optional
    guardrail_version="DRAFT",
    region="us-east-1",
)

result = pipeline.query("What is the refund policy for enterprise customers?")

print(result.answer)
print(result.citations)
print(result.hallucination_risk)  # "low" | "medium" | "high"
```

## Examples

- [`examples/basic_rag.py`](examples/basic_rag.py) — end-to-end pipeline usage
- [`examples/hybrid_search.py`](examples/hybrid_search.py) — hybrid retrieval with RRF scoring

## Architecture

```
query
  │
  ▼
HybridRetriever          ← vector search (Knowledge Base) + keyword search (OpenSearch), merged via RRF
  │
  ▼
ClaudeReranker           ← Claude scores each chunk for relevance, prunes low-signal context
  │
  ▼
Claude (generation)      ← answer generated with citations embedded
  │
  ▼
GuardrailsFilter         ← Bedrock Guardrails checks output before it leaves the pipeline
  │
  ▼
HallucinationDetector    ← Claude checks whether the answer is grounded in retrieved chunks
  │
  ▼
RAGResult
```

## AWS Permissions

The IAM role or user running this library needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock-agent-runtime:Retrieve",
    "bedrock:ApplyGuardrail"
  ],
  "Resource": "*"
}
```

If you're using OpenSearch Serverless for keyword search, add `aoss:APIAccessAll` for your collection.

## License

MIT

