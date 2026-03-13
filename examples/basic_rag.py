"""
basic_rag.py — End-to-end RAG pipeline example
================================================

Demonstrates the full ``RAGPipeline`` with all components enabled:
- Hybrid retrieval (vector + optional keyword search)
- Claude re-ranking
- Claude generation with citations
- Bedrock Guardrails (input + output)
- Hallucination detection

Prerequisites
-------------
1. An Amazon Bedrock Knowledge Base indexed with your documents.
2. (Optional) A Bedrock Guardrail configured for your use case.
3. IAM permissions for bedrock-runtime, bedrock-agent-runtime.
4. Install the library: pip install bedrock-rag-patterns

Configuration
-------------
Set the following environment variables before running:

    export KB_ID="your-knowledge-base-id"
    export GUARDRAIL_ID="your-guardrail-id"   # optional
    export GUARDRAIL_VERSION="DRAFT"           # optional
    export AWS_REGION="us-east-1"
    export AWS_PROFILE="your-profile"          # optional
"""

import logging
import os

from bedrock_rag import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    kb_id = os.environ.get("KB_ID")
    if not kb_id:
        raise SystemExit(
            "Set the KB_ID environment variable to your Bedrock Knowledge Base ID.\n"
            "Example: export KB_ID=ABCDEF1234"
        )

    guardrail_id = os.environ.get("GUARDRAIL_ID")
    guardrail_version = os.environ.get("GUARDRAIL_VERSION", "DRAFT")
    region = os.environ.get("AWS_REGION", "us-east-1")

    print(f"\nInitialising RAGPipeline")
    print(f"  Knowledge Base : {kb_id}")
    print(f"  Guardrail      : {guardrail_id or 'disabled'}")
    print(f"  Region         : {region}\n")

    pipeline = RAGPipeline(
        knowledge_base_id=kb_id,
        guardrail_id=guardrail_id,
        guardrail_version=guardrail_version,
        n_retrieval_results=10,
        rerank=True,
        rerank_min_score=2,
        max_generation_chunks=5,
        hallucination_check=True,
        region=region,
    )

    # --- Example query -------------------------------------------------------
    query = "What is the refund policy for enterprise customers?"

    print(f"Query: {query}\n")
    print("─" * 60)

    result = pipeline.query(query)

    # --- Display answer -------------------------------------------------------
    print("\nAnswer:")
    print(result.answer)

    # --- Citations ------------------------------------------------------------
    if result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for citation in result.citations:
            print(f"  [Source {citation.source_number}] {citation.source_uri}")
            print(f"    \"{citation.chunk_text[:120]}...\"")

    # --- Quality signals -------------------------------------------------------
    print(f"\nHallucination risk : {result.hallucination_risk.upper()}")

    if result.hallucination_detail:
        detail = result.hallucination_detail
        print(f"  Summary          : {detail.summary}")
        print(f"  Claims: {detail.supported_count} supported, "
              f"{detail.inferred_count} inferred, "
              f"{detail.unsupported_count} unsupported")

    if result.input_guardrail_result:
        status = "PASSED" if result.input_guardrail_result.passed else "BLOCKED"
        print(f"\nInput guardrail    : {status}")

    if result.output_guardrail_result:
        status = "PASSED" if result.output_guardrail_result.passed else "BLOCKED"
        print(f"Output guardrail   : {status}")

    # --- Retrieval stats -------------------------------------------------------
    print(f"\nRetrieval stats:")
    print(f"  Retrieved chunks : {len(result.retrieved_chunks)}")
    print(f"  After re-ranking : {len(result.ranked_chunks)}")
    if result.ranked_chunks:
        top = result.ranked_chunks[0]
        print(f"  Top chunk score  : {top.rerank_score}/5 — \"{top.reasoning}\"")


if __name__ == "__main__":
    main()
