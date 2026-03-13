"""
hybrid_search.py — Hybrid retrieval with RRF scoring
=====================================================

Demonstrates the ``HybridRetriever`` in isolation, showing how vector search
and keyword search complement each other and how RRF merges their results.

This example runs two separate retrievals and then a merged retrieval so you
can compare the result sets and understand when each strategy helps.

Prerequisites
-------------
1. A Bedrock Knowledge Base.
2. (For keyword search) An OpenSearch Serverless collection backed by the
   same S3 data as your Knowledge Base.
   Install opensearch-py: pip install bedrock-rag-patterns[opensearch]

Configuration
-------------
    export KB_ID="your-knowledge-base-id"
    export OS_ENDPOINT="https://abc123.us-east-1.aoss.amazonaws.com"  # optional
    export OS_INDEX="bedrock-knowledge-base-default-index"             # optional
    export AWS_REGION="us-east-1"
"""

import logging
import os

from bedrock_rag.client import BedrockRAGClient
from bedrock_rag.retrieval import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def print_results(label: str, chunks: list) -> None:
    print(f"\n{'─' * 60}")
    print(f"{label} ({len(chunks)} results)")
    print("─" * 60)
    for i, chunk in enumerate(chunks, start=1):
        v = f"v={chunk.vector_rank}" if chunk.vector_rank else "v=—"
        k = f"k={chunk.keyword_rank}" if chunk.keyword_rank else "k=—"
        print(f"  #{i:02d}  score={chunk.score:.4f}  [{v} {k}]  {chunk.source_uri}")
        print(f"        {chunk.text[:120].replace(chr(10), ' ')}...")


def main() -> None:
    kb_id = os.environ.get("KB_ID")
    if not kb_id:
        raise SystemExit("Set KB_ID environment variable to your Knowledge Base ID.")

    os_endpoint = os.environ.get("OS_ENDPOINT")
    os_index = os.environ.get("OS_INDEX", "bedrock-knowledge-base-default-index")
    region = os.environ.get("AWS_REGION", "us-east-1")

    client = BedrockRAGClient(region=region)

    # ── Pure vector search (no OpenSearch endpoint) ────────────────────
    vector_only_retriever = HybridRetriever(
        knowledge_base_id=kb_id,
        opensearch_endpoint=None,   # disable keyword leg
        n_results=5,
        client=client,
    )

    # ── Hybrid retriever (vector + keyword) ───────────────────────────
    hybrid_retriever = HybridRetriever(
        knowledge_base_id=kb_id,
        opensearch_endpoint=os_endpoint,
        opensearch_index=os_index,
        n_results=5,
        client=client,
    )

    # ─────────────────────────────────────────────────────────────────
    # Query 1: Semantic query — vector search should do well
    # ─────────────────────────────────────────────────────────────────
    semantic_query = "How does the enterprise subscription pricing work?"
    print(f"\n\nQuery (semantic): {semantic_query}")

    vector_results = vector_only_retriever.retrieve(semantic_query)
    print_results("Vector-only results", vector_results)

    if os_endpoint:
        hybrid_results = hybrid_retriever.retrieve(semantic_query)
        print_results("Hybrid (RRF merged) results", hybrid_results)

    # ─────────────────────────────────────────────────────────────────
    # Query 2: Identifier-based query — keyword search adds value
    # ─────────────────────────────────────────────────────────────────
    identifier_query = "SKU-4892-ENT license terms"
    print(f"\n\nQuery (identifier): {identifier_query}")

    vector_results_id = vector_only_retriever.retrieve(identifier_query)
    print_results("Vector-only results", vector_results_id)

    if os_endpoint:
        hybrid_results_id = hybrid_retriever.retrieve(identifier_query)
        print_results("Hybrid (RRF merged) results", hybrid_results_id)

    # ─────────────────────────────────────────────────────────────────
    # Demonstrate RRF boosting for a document that appears in both sets
    # ─────────────────────────────────────────────────────────────────
    if os_endpoint:
        print("\n\nRRF score breakdown for hybrid results:")
        print("─" * 60)
        print(
            f"{'#':<4} {'RRF score':<12} {'Vector rank':<14} "
            f"{'Keyword rank':<14} {'Source'}"
        )
        print("─" * 60)
        for i, chunk in enumerate(hybrid_results[:5], start=1):
            vr = str(chunk.vector_rank) if chunk.vector_rank else "—"
            kr = str(chunk.keyword_rank) if chunk.keyword_rank else "—"
            src = (chunk.source_uri or "unknown")[-50:]
            print(f"{i:<4} {chunk.score:<12.4f} {vr:<14} {kr:<14} {src}")

        print("\nNote: Documents appearing in both result sets (vector_rank AND")
        print("keyword_rank populated) receive a higher RRF score due to dual")
        print("ranking signals.")


if __name__ == "__main__":
    main()
