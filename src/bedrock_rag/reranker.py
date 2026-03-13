"""
ClaudeReranker
==============

Uses Claude to re-rank retrieved chunks by relevance to a query.

Why re-rank?
------------
Retrieval systems optimise for recall — they return chunks that are plausibly
relevant.  But a chunk that is topically adjacent to the query is not the same
as a chunk that *answers* it.  A cross-encoder re-ranker closes this gap.

This implementation uses Claude as a cross-encoder: for each chunk, Claude is
asked to judge how well it answers the query on a scale of 1–5, with a brief
reasoning trace.  Chunks are then sorted by that score and optionally pruned
below a threshold.

Trade-offs
----------
- **Latency**: Scoring N chunks requires N Claude API calls (or one batched
  prompt).  This implementation uses a single batched prompt listing all
  chunks, which is faster but uses more tokens.
- **Cost**: Re-ranking adds inference cost.  Use Claude Haiku for speed-
  sensitive paths; use Sonnet for quality-sensitive paths.
- **Threshold**: The default minimum score of 2/5 aggressively prunes
  off-topic chunks.  Tune ``min_score`` for your recall/precision trade-off.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from bedrock_rag.client import BedrockRAGClient, DEFAULT_CLAUDE_MODEL_ID
from bedrock_rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = """\
You are a precise relevance judge for a retrieval-augmented generation system.
Your job is to score how well each retrieved text chunk answers the user's query.

Scoring rubric:
  5 — Directly and completely answers the query.
  4 — Mostly answers the query; minor gaps.
  3 — Partially relevant; contains useful information but incomplete.
  2 — Tangentially related; unlikely to help answer the query.
  1 — Irrelevant.

Respond ONLY with a JSON array. Each element must have:
  "index"     : integer (the chunk index from the input)
  "score"     : integer 1–5
  "reasoning" : one sentence explaining the score

Do not include any text outside the JSON array."""

_RERANK_USER_TEMPLATE = """\
Query: {query}

Chunks to score:
{chunks_text}

Return a JSON array with one object per chunk."""


@dataclass
class RankedChunk:
    """A retrieved chunk with a re-ranking score.

    Attributes
    ----------
    chunk:
        The original ``RetrievedChunk``.
    rerank_score:
        Claude's relevance score (1–5).
    reasoning:
        Claude's one-sentence explanation.
    """

    chunk: RetrievedChunk
    rerank_score: int
    reasoning: str


class ClaudeReranker:
    """Re-ranks retrieved chunks using Claude.

    Parameters
    ----------
    model_id:
        Bedrock model ID to use for scoring.  Claude Haiku is fast and cheap
        for this task; Sonnet gives marginally better discrimination.
    min_score:
        Chunks with a re-rank score strictly below this value are dropped.
        Set to 1 to disable pruning.
    client:
        ``BedrockRAGClient`` instance.
    region:
        AWS region, used only if ``client`` is not provided.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CLAUDE_MODEL_ID,
        min_score: int = 2,
        client: BedrockRAGClient | None = None,
        region: str = "us-east-1",
    ) -> None:
        if not 1 <= min_score <= 5:
            raise ValueError("min_score must be between 1 and 5")
        self.model_id = model_id
        self.min_score = min_score
        self.client = client or BedrockRAGClient(region=region)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_chunks: int = 20,
    ) -> list[RankedChunk]:
        """Re-rank ``chunks`` by relevance to ``query``.

        Parameters
        ----------
        query:
            The user's query.
        chunks:
            Retrieved chunks to score.  Typically the output of
            ``HybridRetriever.retrieve()``.
        max_chunks:
            Maximum number of chunks to send to Claude in a single prompt.
            Chunks beyond this limit are dropped before scoring to avoid
            context window overflow.

        Returns
        -------
        list[RankedChunk]
            Chunks that passed the ``min_score`` threshold, sorted by
            ``rerank_score`` descending then original RRF score descending.
        """
        if not chunks:
            return []

        # Truncate to avoid hitting context limits
        working = chunks[:max_chunks]
        logger.debug("Re-ranking %d chunks for query: %.80s", len(working), query)

        chunks_text = self._format_chunks(working)
        prompt = _RERANK_USER_TEMPLATE.format(query=query, chunks_text=chunks_text)

        raw_response = self.client.invoke_model(
            model_id=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            system=_RERANK_SYSTEM,
            max_tokens=1024,
            temperature=0.0,
        )

        scores = self._parse_scores(raw_response, len(working))

        ranked: list[RankedChunk] = []
        for item in scores:
            idx = item["index"]
            if idx < 0 or idx >= len(working):
                logger.warning("Re-ranker returned out-of-range index %d; skipping", idx)
                continue
            score = int(item.get("score", 1))
            if score < self.min_score:
                logger.debug("Chunk %d pruned (score %d < min_score %d)", idx, score, self.min_score)
                continue
            ranked.append(
                RankedChunk(
                    chunk=working[idx],
                    rerank_score=score,
                    reasoning=str(item.get("reasoning", "")),
                )
            )

        ranked.sort(key=lambda r: (r.rerank_score, r.chunk.score), reverse=True)
        logger.info(
            "Re-ranker kept %d/%d chunks (min_score=%d)",
            len(ranked),
            len(working),
            self.min_score,
        )
        return ranked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        """Format chunks as a numbered list for the scoring prompt."""
        lines = []
        for i, chunk in enumerate(chunks):
            # Truncate very long chunks to keep the prompt manageable
            text = chunk.text[:800].replace("\n", " ")
            lines.append(f"[{i}] {text}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_scores(response: str, expected_count: int) -> list[dict[str, Any]]:
        """Extract the JSON array from Claude's response.

        Claude is instructed to return only JSON, but it occasionally wraps
        the array in a markdown code block.  This method strips that wrapper.
        """
        # Strip optional markdown code fence
        clean = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error(
                "Re-ranker response was not valid JSON (expected %d scores): %s\nResponse: %.500s",
                expected_count,
                exc,
                response,
            )
            # Graceful degradation: return all chunks with score 1 so the
            # pipeline can continue rather than crashing.
            return [{"index": i, "score": 1, "reasoning": "parse error"} for i in range(expected_count)]

        if not isinstance(parsed, list):
            logger.error("Re-ranker response was not a JSON array; degrading gracefully")
            return [{"index": i, "score": 1, "reasoning": "unexpected format"} for i in range(expected_count)]

        return parsed
