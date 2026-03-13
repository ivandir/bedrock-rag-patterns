"""
bedrock-rag-patterns
====================

Production-ready RAG patterns on AWS Bedrock.

Quickstart
----------
    from bedrock_rag import RAGPipeline

    pipeline = RAGPipeline(
        knowledge_base_id="ABCDEF1234",
        guardrail_id="gr-abc123",       # optional
        guardrail_version="DRAFT",
    )
    result = pipeline.query("What is the refund policy?")
    print(result.answer)

Classes
-------
- :class:`~bedrock_rag.client.BedrockRAGClient` — thin boto3 wrapper
- :class:`~bedrock_rag.retrieval.HybridRetriever` — vector + keyword search with RRF
- :class:`~bedrock_rag.reranker.ClaudeReranker` — Claude-based relevance re-ranking
- :class:`~bedrock_rag.guardrails.GuardrailsFilter` — Bedrock Guardrails integration
- :class:`~bedrock_rag.hallucination.HallucinationDetector` — grounding verification
- :class:`~bedrock_rag.pipeline.RAGPipeline` — full orchestrated pipeline
"""

from bedrock_rag.client import BedrockRAGClient
from bedrock_rag.guardrails import GuardrailsFilter, GuardrailsResult
from bedrock_rag.hallucination import HallucinationDetector, HallucinationResult
from bedrock_rag.pipeline import RAGPipeline, RAGResult
from bedrock_rag.reranker import ClaudeReranker, RankedChunk
from bedrock_rag.retrieval import HybridRetriever, RetrievedChunk

__all__ = [
    "BedrockRAGClient",
    "HybridRetriever",
    "RetrievedChunk",
    "ClaudeReranker",
    "RankedChunk",
    "GuardrailsFilter",
    "GuardrailsResult",
    "HallucinationDetector",
    "HallucinationResult",
    "RAGPipeline",
    "RAGResult",
]

__version__ = "0.1.0"
