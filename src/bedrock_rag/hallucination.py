"""
HallucinationDetector
=====================

Uses Claude to verify that a generated answer is grounded in the retrieved
context chunks.

The problem
-----------
RAG does not eliminate hallucination; it reduces it.  A model given 10 chunks
may still:

- Interpolate information across chunks in a way that is not supported by any
  single chunk.
- Confidently cite a number, date, or name that does not appear in the context.
- State a correct-sounding fact that contradicts the provided source material.

Approach
--------
We ask Claude to act as a fact-checker: given the generated answer and the
source chunks, identify every claim in the answer and classify it as:

- ``SUPPORTED``  — the claim is directly supported by at least one chunk.
- ``INFERRED``   — the claim follows logically from the chunks but is not
                   stated verbatim.  Low risk but worth flagging.
- ``UNSUPPORTED`` — the claim is not present in and cannot be inferred from
                   the chunks.  This is a hallucination.

The overall hallucination risk is derived from the ratio of unsupported claims.

This is not a perfect detector — it is itself subject to Claude's own
understanding — but it catches the most egregious fabrications reliably.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from bedrock_rag.client import BedrockRAGClient, DEFAULT_CLAUDE_MODEL_ID
from bedrock_rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

HallucinationRisk = Literal["low", "medium", "high"]

_DETECTOR_SYSTEM = """\
You are a meticulous fact-checker for a retrieval-augmented generation system.
Your task is to verify that every factual claim in an AI-generated answer is
supported by the provided source chunks.

Instructions:
1. Break the answer into individual factual claims.
2. For each claim, determine whether it is:
   - SUPPORTED: directly stated in at least one source chunk.
   - INFERRED: logically follows from the source chunks but not verbatim.
   - UNSUPPORTED: not present in and not inferable from the source chunks.
3. Return ONLY a JSON object with this exact structure:
   {
     "claims": [
       {
         "claim": "the claim text",
         "status": "SUPPORTED" | "INFERRED" | "UNSUPPORTED",
         "evidence": "quote from source chunk, or empty string if UNSUPPORTED"
       }
     ],
     "risk": "low" | "medium" | "high",
     "summary": "one sentence summary of hallucination risk"
   }

Risk level rules:
  low    — 0 UNSUPPORTED claims.
  medium — 1–2 UNSUPPORTED claims or more than 30% INFERRED.
  high   — 3+ UNSUPPORTED claims or any UNSUPPORTED claim that is a key fact.

Do not include any text outside the JSON object."""

_DETECTOR_USER_TEMPLATE = """\
Answer to verify:
{answer}

Source chunks:
{chunks_text}"""


@dataclass
class ClaimVerification:
    """Verification result for a single factual claim.

    Attributes
    ----------
    claim:
        The extracted claim text.
    status:
        ``"SUPPORTED"``, ``"INFERRED"``, or ``"UNSUPPORTED"``.
    evidence:
        A quote from the source chunks supporting the claim, or an empty
        string for unsupported claims.
    """

    claim: str
    status: Literal["SUPPORTED", "INFERRED", "UNSUPPORTED"]
    evidence: str = ""


@dataclass
class HallucinationResult:
    """Result of a hallucination detection check.

    Attributes
    ----------
    risk:
        Overall risk level: ``"low"``, ``"medium"``, or ``"high"``.
    summary:
        One-sentence summary from Claude.
    claims:
        Per-claim verification results.
    unsupported_count:
        Number of claims flagged as ``UNSUPPORTED``.
    inferred_count:
        Number of claims flagged as ``INFERRED``.
    supported_count:
        Number of claims flagged as ``SUPPORTED``.
    """

    risk: HallucinationRisk
    summary: str
    claims: list[ClaimVerification] = field(default_factory=list)
    unsupported_count: int = 0
    inferred_count: int = 0
    supported_count: int = 0

    @property
    def passed(self) -> bool:
        """``True`` if risk is ``"low"``."""
        return self.risk == "low"


class HallucinationDetector:
    """Detect hallucinations in a generated answer.

    Parameters
    ----------
    model_id:
        Claude model to use.  Sonnet is recommended for this task because
        claim-level fact-checking requires careful reading.
    max_context_chunks:
        Maximum number of source chunks to include in the detection prompt.
        Exceeding Claude's context window will cause an API error.
    client:
        ``BedrockRAGClient`` instance.
    region:
        AWS region, used only if ``client`` is not provided.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CLAUDE_MODEL_ID,
        max_context_chunks: int = 10,
        client: BedrockRAGClient | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.model_id = model_id
        self.max_context_chunks = max_context_chunks
        self.client = client or BedrockRAGClient(region=region)

    def check(
        self,
        answer: str,
        context_chunks: list[RetrievedChunk],
    ) -> HallucinationResult:
        """Check whether ``answer`` is grounded in ``context_chunks``.

        Parameters
        ----------
        answer:
            The generated answer to verify.
        context_chunks:
            The source chunks that were used to generate the answer.  Pass
            the re-ranked list if available; otherwise the raw retrieval list.

        Returns
        -------
        HallucinationResult
            Structured result with per-claim verdicts and an overall risk level.
        """
        if not answer.strip():
            logger.warning("HallucinationDetector received an empty answer")
            return HallucinationResult(risk="low", summary="No answer to verify.")

        if not context_chunks:
            logger.warning("HallucinationDetector received no context chunks")
            return HallucinationResult(
                risk="high",
                summary="No source context provided; cannot verify any claims.",
            )

        working_chunks = context_chunks[: self.max_context_chunks]
        chunks_text = self._format_chunks(working_chunks)

        prompt = _DETECTOR_USER_TEMPLATE.format(
            answer=answer,
            chunks_text=chunks_text,
        )

        raw_response = self.client.invoke_model(
            model_id=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            system=_DETECTOR_SYSTEM,
            max_tokens=2048,
            temperature=0.0,
        )

        return self._parse_response(raw_response)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.text[:1000].replace("\n", " ")
            source = chunk.source_uri or f"chunk-{i}"
            lines.append(f"[Source {i}: {source}]\n{text}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_response(response: str) -> HallucinationResult:
        clean = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

        try:
            parsed: dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error(
                "HallucinationDetector response was not valid JSON: %s\nResponse: %.500s",
                exc,
                response,
            )
            return HallucinationResult(
                risk="medium",
                summary="Could not parse hallucination check response; treating as medium risk.",
            )

        claims_raw = parsed.get("claims", [])
        claims: list[ClaimVerification] = []
        supported = inferred = unsupported = 0

        for c in claims_raw:
            status = str(c.get("status", "UNSUPPORTED")).upper()
            if status not in ("SUPPORTED", "INFERRED", "UNSUPPORTED"):
                status = "UNSUPPORTED"

            claim = ClaimVerification(
                claim=str(c.get("claim", "")),
                status=status,  # type: ignore[arg-type]
                evidence=str(c.get("evidence", "")),
            )
            claims.append(claim)

            if status == "SUPPORTED":
                supported += 1
            elif status == "INFERRED":
                inferred += 1
            else:
                unsupported += 1

        raw_risk = str(parsed.get("risk", "medium")).lower()
        if raw_risk not in ("low", "medium", "high"):
            raw_risk = "medium"

        return HallucinationResult(
            risk=raw_risk,  # type: ignore[arg-type]
            summary=str(parsed.get("summary", "")),
            claims=claims,
            supported_count=supported,
            inferred_count=inferred,
            unsupported_count=unsupported,
        )
