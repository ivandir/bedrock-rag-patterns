"""
BedrockRAGClient
================

Thin wrapper around the two boto3 clients needed for RAG on Bedrock:

- ``bedrock-runtime``        — InvokeModel, InvokeModelWithResponseStream, ApplyGuardrail
- ``bedrock-agent-runtime``  — Retrieve (Knowledge Bases)

All other classes in this library accept a ``BedrockRAGClient`` (or None, in
which case they construct one with defaults).  Centralising session/client
creation here makes it easy to inject mocked clients in tests and to share a
single session across a pipeline invocation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Default model IDs — all available in us-east-1
DEFAULT_CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_REGION = "us-east-1"

# Retry configuration: exponential back-off up to 3 retries to handle
# throttling from the Bedrock service limits.
_RETRY_CONFIG = Config(
    retries={
        "max_attempts": 3,
        "mode": "adaptive",
    }
)


class BedrockRAGClient:
    """Shared boto3 clients for Bedrock RAG workloads.

    Parameters
    ----------
    region:
        AWS region.  Bedrock Knowledge Bases are regional, so all resources
        (Knowledge Base, Guardrail, OpenSearch collection) must be in the same
        region.
    profile_name:
        Optional AWS profile name.  Useful for local development; leave as
        ``None`` in Lambda / ECS where credentials come from the task role.
    session:
        Bring-your-own boto3 Session.  When provided, ``region`` and
        ``profile_name`` are ignored.
    """

    def __init__(
        self,
        region: str = DEFAULT_REGION,
        profile_name: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        self.region = region

        if session is not None:
            self._session = session
        else:
            self._session = boto3.Session(
                region_name=region,
                profile_name=profile_name,
            )

        self._bedrock_runtime: Any = None
        self._bedrock_agent_runtime: Any = None

    # ------------------------------------------------------------------
    # Lazy client properties — only create clients when first needed so
    # that the constructor stays fast and testable.
    # ------------------------------------------------------------------

    @property
    def bedrock_runtime(self) -> Any:
        """boto3 client for ``bedrock-runtime``."""
        if self._bedrock_runtime is None:
            self._bedrock_runtime = self._session.client(
                service_name="bedrock-runtime",
                config=_RETRY_CONFIG,
            )
        return self._bedrock_runtime

    @property
    def bedrock_agent_runtime(self) -> Any:
        """boto3 client for ``bedrock-agent-runtime`` (Knowledge Bases)."""
        if self._bedrock_agent_runtime is None:
            self._bedrock_agent_runtime = self._session.client(
                service_name="bedrock-agent-runtime",
                config=_RETRY_CONFIG,
            )
        return self._bedrock_agent_runtime

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def invoke_model(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """Call ``bedrock-runtime:InvokeModel`` with a Claude Messages payload.

        Parameters
        ----------
        model_id:
            Bedrock model ID, e.g. ``anthropic.claude-3-5-sonnet-20241022-v2:0``.
        messages:
            List of ``{"role": "user"|"assistant", "content": str}`` dicts.
        system:
            Optional system prompt.
        max_tokens:
            Maximum tokens in the response.
        temperature:
            Sampling temperature.  Use 0.0 for deterministic classification
            tasks (re-ranking, hallucination detection).

        Returns
        -------
        str
            The text content of the first message block in the response.

        Raises
        ------
        ClientError
            On Bedrock API errors (throttling, model not available, etc.).
        """
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except ClientError as exc:
            logger.error(
                "bedrock:InvokeModel failed for model %s: %s",
                model_id,
                exc.response["Error"]["Message"],
            )
            raise

        raw = json.loads(response["body"].read())
        # Claude Messages API response shape
        return str(raw["content"][0]["text"])

    def retrieve(
        self,
        knowledge_base_id: str,
        query: str,
        n_results: int = 10,
        filter_expression: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Call ``bedrock-agent-runtime:Retrieve`` on a Knowledge Base.

        Parameters
        ----------
        knowledge_base_id:
            The Bedrock Knowledge Base ID.
        query:
            Natural-language query.
        n_results:
            Number of chunks to return.
        filter_expression:
            Optional metadata filter in the Bedrock RetrievalFilter format.

        Returns
        -------
        list[dict]
            List of ``RetrievalResult`` dicts from the Bedrock API.
        """
        config: dict[str, Any] = {
            "vectorSearchConfiguration": {
                "numberOfResults": n_results,
            }
        }
        if filter_expression:
            config["vectorSearchConfiguration"]["filter"] = filter_expression

        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=config,
            )
        except ClientError as exc:
            logger.error(
                "bedrock-agent-runtime:Retrieve failed for KB %s: %s",
                knowledge_base_id,
                exc.response["Error"]["Message"],
            )
            raise

        return list(response.get("retrievalResults", []))

    def apply_guardrail(
        self,
        guardrail_id: str,
        guardrail_version: str,
        source: str,
        content: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call ``bedrock-runtime:ApplyGuardrail``.

        Parameters
        ----------
        guardrail_id:
            Bedrock Guardrail ID.
        guardrail_version:
            Guardrail version string, e.g. ``"DRAFT"`` or ``"1"``.
        source:
            ``"INPUT"`` or ``"OUTPUT"``.
        content:
            List of content blocks, e.g.
            ``[{"text": {"text": "the text to check"}}]``.

        Returns
        -------
        dict
            Full API response including ``action`` (``"NONE"`` or
            ``"GUARDRAIL_INTERVENED"``).
        """
        try:
            return dict(
                self.bedrock_runtime.apply_guardrail(
                    guardrailIdentifier=guardrail_id,
                    guardrailVersion=guardrail_version,
                    source=source,
                    content=content,
                )
            )
        except ClientError as exc:
            logger.error(
                "bedrock:ApplyGuardrail failed for guardrail %s: %s",
                guardrail_id,
                exc.response["Error"]["Message"],
            )
            raise
