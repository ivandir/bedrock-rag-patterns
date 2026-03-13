"""
GuardrailsFilter
================

Wraps the Bedrock Guardrails ``ApplyGuardrail`` API for input and output
filtering.

What Bedrock Guardrails covers
------------------------------
- **Content filtering** — harmful, hateful, or violent content.
- **Denied topics** — configurable topic blocklist (e.g., "do not discuss
  competitor products").
- **PII redaction** — detects and optionally redacts personally identifiable
  information before it reaches the model or the end user.
- **Word filters** — exact-match blocklist for profanity or sensitive terms.
- **Grounding check** — Bedrock's built-in factual grounding check (separate
  from our custom HallucinationDetector).

Pipeline placement
------------------
Apply Guardrails twice:

1. **On the user query** (source="INPUT") — catch prompt injection, jailbreak
   attempts, and queries on denied topics before spending inference budget.
2. **On the generated answer** (source="OUTPUT") — catch PII leakage and
   content policy violations in the model's response.

The ``GuardrailsFilter`` class handles both cases with a single ``check()``
method driven by the ``source`` parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from bedrock_rag.client import BedrockRAGClient

logger = logging.getLogger(__name__)

GuardrailSource = Literal["INPUT", "OUTPUT"]


@dataclass
class GuardrailsResult:
    """Result of a guardrail check.

    Attributes
    ----------
    passed:
        ``True`` if the guardrail did not intervene (action == "NONE").
    action:
        Raw action string from the API: ``"NONE"`` or
        ``"GUARDRAIL_INTERVENED"``.
    interventions:
        List of intervention detail dicts from the API response.
    output_text:
        The (possibly redacted) text returned by the guardrail when it
        intervenes.  Empty string if the guardrail did not intervene.
    raw_response:
        Full API response dict.
    """

    passed: bool
    action: str
    interventions: list[dict[str, Any]] = field(default_factory=list)
    output_text: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


class GuardrailsFilter:
    """Apply Bedrock Guardrails to text.

    Parameters
    ----------
    guardrail_id:
        Bedrock Guardrail ID.
    guardrail_version:
        Guardrail version, e.g. ``"DRAFT"`` or ``"1"``.
    raise_on_intervention:
        If ``True``, raise a ``GuardrailInterventionError`` when the guardrail
        blocks content.  If ``False`` (default), return a ``GuardrailsResult``
        with ``passed=False`` and let the caller decide.
    client:
        ``BedrockRAGClient`` instance.
    region:
        AWS region, used only if ``client`` is not provided.
    """

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str = "DRAFT",
        raise_on_intervention: bool = False,
        client: BedrockRAGClient | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.raise_on_intervention = raise_on_intervention
        self.client = client or BedrockRAGClient(region=region)

    def check(
        self,
        text: str,
        source: GuardrailSource = "OUTPUT",
    ) -> GuardrailsResult:
        """Run a guardrail check on ``text``.

        Parameters
        ----------
        text:
            The text to check.
        source:
            ``"INPUT"`` for user queries; ``"OUTPUT"`` for generated answers.

        Returns
        -------
        GuardrailsResult
            Result object.  Check ``result.passed`` to determine whether to
            proceed.

        Raises
        ------
        GuardrailInterventionError
            If ``raise_on_intervention=True`` and the guardrail blocks content.
        """
        content = [{"text": {"text": text}}]

        response = self.client.apply_guardrail(
            guardrail_id=self.guardrail_id,
            guardrail_version=self.guardrail_version,
            source=source,
            content=content,
        )

        action = response.get("action", "NONE")
        passed = action == "NONE"

        # Extract intervention details (may be empty list)
        interventions: list[dict[str, Any]] = []
        for assessment in response.get("assessments", []):
            for category in ("contentPolicy", "topicPolicy", "wordPolicy", "sensitiveInformationPolicy"):
                policy_result = assessment.get(category, {})
                if policy_result:
                    interventions.append({category: policy_result})

        # The guardrail may return sanitised/redacted output text
        output_text = ""
        for output_block in response.get("output", []):
            output_text += output_block.get("text", "")

        logger.info(
            "Guardrail %s (%s) action=%s on %s text (%d chars)",
            self.guardrail_id,
            self.guardrail_version,
            action,
            source,
            len(text),
        )
        if not passed:
            logger.warning(
                "Guardrail intervened: %d intervention(s) detected", len(interventions)
            )

        result = GuardrailsResult(
            passed=passed,
            action=action,
            interventions=interventions,
            output_text=output_text,
            raw_response=response,
        )

        if not passed and self.raise_on_intervention:
            raise GuardrailInterventionError(
                f"Bedrock Guardrail {self.guardrail_id} blocked content "
                f"(source={source}, action={action})",
                result=result,
            )

        return result

    def check_input(self, text: str) -> GuardrailsResult:
        """Convenience wrapper for ``check(text, source="INPUT")``."""
        return self.check(text, source="INPUT")

    def check_output(self, text: str) -> GuardrailsResult:
        """Convenience wrapper for ``check(text, source="OUTPUT")``."""
        return self.check(text, source="OUTPUT")


class GuardrailInterventionError(Exception):
    """Raised when a guardrail blocks content and ``raise_on_intervention=True``.

    Attributes
    ----------
    result:
        The ``GuardrailsResult`` that triggered the exception.
    """

    def __init__(self, message: str, result: GuardrailsResult) -> None:
        super().__init__(message)
        self.result = result
