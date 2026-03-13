"""
RAGPipeline
===========

Orchestrates the full RAG pipeline:

    query
      → input guardrail check (optional)
      → hybrid retrieval (vector + keyword, RRF merged)
      → Claude re-ranking (prunes low-relevance chunks)
      → Claude generation (with citation instructions)
      → output guardrail check (optional)
      → hallucination detection
      → RAGResult

Design decisions
----------------
- **Guardrails first on input**: Fail fast on denied topics and prompt
  injection before spending retrieval and generation budget.
- **Guardrails on output**: A second check after generation catches PII
  that leaked through from the source documents into the answer.
- **Hallucination check last**: We check the final answer against the
  retrieved chunks so the grounding check uses the same context the model
  used to generate the answer.
- **Citations in the generation prompt**: The generation prompt instructs
  Claude to cite sources using ``[Source N]`` notation.  The pipeline
  extracts these citations into ``RAGResult.citations``.

Skipping steps
--------------
All pipeline components are optional:

- Omit ``guardrail_id`` to skip guardrail checks.
- Set ``rerank=False`` to skip re-ranking.
- Set ``hallucination_check=False`` to skip hallucination detection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bedrock_rag.client import BedrockRAGClient, DEFAULT_CLAUDE_MODEL_ID
from bedrock_rag.guardrails import GuardrailInterventionError, GuardrailsFilter, GuardrailsResult
from bedrock_rag.hallucination import HallucinationDetector, HallucinationResult, HallucinationRisk
from bedrock_rag.reranker import ClaudeReranker, RankedChunk
from bedrock_rag.retrieval import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)

_GENERATION_SYSTEM = """\
You are a helpful assistant that answers questions using only the provided
source documents.

Rules:
1. Base your answer exclusively on the provided source documents.
2. Cite sources using [Source N] notation immediately after each claim, where N
   matches the source number in the list provided.
3. If the source documents do not contain enough information to answer the
   question, say so clearly rather than guessing.
4. Do not make up information that is not in the sources.
5. Be concise and direct."""

_GENERATION_USER_TEMPLATE = """\
Sources:
{sources}

Question: {query}

Answer (cite sources inline):"""


@dataclass
class Citation:
    """A source citation extracted from the generated answer.

    Attributes
    ----------
    source_number:
        The ``[Source N]`` number referenced in the answer.
    source_uri:
        The S3 URI or other location of the source document.
    chunk_text:
        The text of the chunk that was cited.
    """

    source_number: int
    source_uri: str
    chunk_text: str


@dataclass
class RAGResult:
    """The result of a full RAG pipeline invocation.

    Attributes
    ----------
    answer:
        The generated answer text (with ``[Source N]`` citations inline).
    citations:
        Structured citation objects corresponding to ``[Source N]`` references
        in the answer.
    hallucination_risk:
        Overall hallucination risk: ``"low"``, ``"medium"``, or ``"high"``.
    hallucination_detail:
        Full hallucination detection result.  ``None`` if hallucination
        detection was skipped.
    input_guardrail_result:
        Result of the input guardrail check.  ``None`` if no guardrail was
        configured.
    output_guardrail_result:
        Result of the output guardrail check.  ``None`` if no guardrail was
        configured.
    retrieved_chunks:
        Chunks returned by the retriever before re-ranking.
    ranked_chunks:
        Chunks after re-ranking.  Same as ``retrieved_chunks`` (wrapped in
        ``RankedChunk``) if re-ranking was skipped.
    query:
        The original query.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    hallucination_risk: HallucinationRisk = "low"
    hallucination_detail: HallucinationResult | None = None
    input_guardrail_result: GuardrailsResult | None = None
    output_guardrail_result: GuardrailsResult | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    ranked_chunks: list[RankedChunk] = field(default_factory=list)
    query: str = ""


class RAGPipeline:
    """Full RAG pipeline on AWS Bedrock.

    Parameters
    ----------
    knowledge_base_id:
        Bedrock Knowledge Base ID for vector retrieval.
    generation_model_id:
        Claude model used for answer generation.
    reranking_model_id:
        Claude model used for re-ranking.  Defaults to the same as
        ``generation_model_id``.
    opensearch_endpoint:
        Optional OpenSearch endpoint for keyword search leg.
    opensearch_index:
        OpenSearch index name.
    guardrail_id:
        Bedrock Guardrail ID.  If ``None``, guardrail checks are skipped.
    guardrail_version:
        Guardrail version string.
    n_retrieval_results:
        Number of results to fetch from each retriever before merging.
    rerank:
        Whether to re-rank retrieved chunks with Claude.
    rerank_min_score:
        Minimum re-rank score to keep a chunk (1–5).
    max_generation_chunks:
        Maximum number of chunks to include in the generation prompt.
    hallucination_check:
        Whether to run hallucination detection on the generated answer.
    client:
        Shared ``BedrockRAGClient``.  If not provided, one is created.
    region:
        AWS region.
    """

    def __init__(
        self,
        knowledge_base_id: str,
        generation_model_id: str = DEFAULT_CLAUDE_MODEL_ID,
        reranking_model_id: str | None = None,
        opensearch_endpoint: str | None = None,
        opensearch_index: str = "bedrock-knowledge-base-default-index",
        guardrail_id: str | None = None,
        guardrail_version: str = "DRAFT",
        n_retrieval_results: int = 10,
        rerank: bool = True,
        rerank_min_score: int = 2,
        max_generation_chunks: int = 5,
        hallucination_check: bool = True,
        client: BedrockRAGClient | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.client = client or BedrockRAGClient(region=region)
        self.generation_model_id = generation_model_id
        self.max_generation_chunks = max_generation_chunks
        self.hallucination_check_enabled = hallucination_check

        self.retriever = HybridRetriever(
            knowledge_base_id=knowledge_base_id,
            opensearch_endpoint=opensearch_endpoint,
            opensearch_index=opensearch_index,
            n_results=n_retrieval_results,
            client=self.client,
        )

        self.reranker: ClaudeReranker | None = None
        if rerank:
            self.reranker = ClaudeReranker(
                model_id=reranking_model_id or generation_model_id,
                min_score=rerank_min_score,
                client=self.client,
            )

        self.guardrails: GuardrailsFilter | None = None
        if guardrail_id:
            self.guardrails = GuardrailsFilter(
                guardrail_id=guardrail_id,
                guardrail_version=guardrail_version,
                raise_on_intervention=False,
                client=self.client,
            )

        self.hallucination_detector: HallucinationDetector | None = None
        if hallucination_check:
            self.hallucination_detector = HallucinationDetector(
                model_id=generation_model_id,
                client=self.client,
            )

    def query(
        self,
        query: str,
        filter_expression: dict[str, Any] | None = None,
    ) -> RAGResult:
        """Run the full RAG pipeline for a query.

        Parameters
        ----------
        query:
            The user's question.
        filter_expression:
            Optional Bedrock metadata filter for the retrieval step.

        Returns
        -------
        RAGResult
            Full structured result including answer, citations, and quality
            signals.
        """
        result = RAGResult(query=query)
        logger.info("RAGPipeline query: %.120s", query)

        # ── Step 1: Input guardrail check ─────────────────────────────
        if self.guardrails:
            input_result = self.guardrails.check_input(query)
            result.input_guardrail_result = input_result
            if not input_result.passed:
                logger.warning("Input guardrail blocked query; returning early")
                result.answer = (
                    "I'm sorry, I can't help with that request."
                    if not input_result.output_text
                    else input_result.output_text
                )
                result.hallucination_risk = "low"
                return result

        # ── Step 2: Retrieval ─────────────────────────────────────────
        retrieved = self.retriever.retrieve(query, filter_expression=filter_expression)
        result.retrieved_chunks = retrieved

        if not retrieved:
            logger.warning("No chunks retrieved for query")
            result.answer = "I could not find any relevant information to answer your question."
            result.hallucination_risk = "low"
            return result

        # ── Step 3: Re-ranking ────────────────────────────────────────
        if self.reranker:
            ranked = self.reranker.rerank(query, retrieved)
        else:
            # Wrap raw chunks in RankedChunk with a neutral score
            ranked = [
                RankedChunk(chunk=c, rerank_score=3, reasoning="re-ranking disabled")
                for c in retrieved
            ]
        result.ranked_chunks = ranked

        if not ranked:
            logger.warning("All chunks pruned by re-ranker; returning no-answer response")
            result.answer = "The retrieved documents did not contain information relevant enough to answer your question."
            result.hallucination_risk = "low"
            return result

        # ── Step 4: Generation ────────────────────────────────────────
        generation_chunks = [rc.chunk for rc in ranked[: self.max_generation_chunks]]
        answer, citations = self._generate(query, generation_chunks)
        result.answer = answer
        result.citations = citations

        # ── Step 5: Output guardrail check ────────────────────────────
        if self.guardrails:
            output_result = self.guardrails.check_output(answer)
            result.output_guardrail_result = output_result
            if not output_result.passed:
                logger.warning("Output guardrail blocked answer; returning safe message")
                result.answer = (
                    "The generated response was blocked by content safety filters."
                    if not output_result.output_text
                    else output_result.output_text
                )
                result.citations = []

        # ── Step 6: Hallucination detection ───────────────────────────
        if self.hallucination_detector:
            hall_result = self.hallucination_detector.check(result.answer, generation_chunks)
            result.hallucination_detail = hall_result
            result.hallucination_risk = hall_result.risk
            if hall_result.risk == "high":
                logger.warning(
                    "High hallucination risk detected (%d unsupported claims): %s",
                    hall_result.unsupported_count,
                    hall_result.summary,
                )

        logger.info(
            "RAGPipeline complete: %d citations, hallucination_risk=%s",
            len(result.citations),
            result.hallucination_risk,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[Citation]]:
        """Generate an answer from ``chunks`` for ``query``.

        Returns
        -------
        tuple[str, list[Citation]]
            The answer text and extracted citations.
        """
        sources_text = self._format_sources(chunks)
        prompt = _GENERATION_USER_TEMPLATE.format(sources=sources_text, query=query)

        answer = self.client.invoke_model(
            model_id=self.generation_model_id,
            messages=[{"role": "user", "content": prompt}],
            system=_GENERATION_SYSTEM,
            max_tokens=2048,
            temperature=0.1,
        )

        citations = self._extract_citations(answer, chunks)
        return answer, citations

    @staticmethod
    def _format_sources(chunks: list[RetrievedChunk]) -> str:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f"[Source {i}: {chunk.source_uri or 'unknown'}]\n{chunk.text}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        """Extract ``[Source N]`` references from the generated answer."""
        cited_numbers = {int(n) for n in re.findall(r"\[Source\s+(\d+)\]", answer)}
        citations: list[Citation] = []
        for n in sorted(cited_numbers):
            idx = n - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                citations.append(
                    Citation(
                        source_number=n,
                        source_uri=chunk.source_uri,
                        chunk_text=chunk.text[:300],
                    )
                )
        return citations
