"""
HybridRetriever
===============

Combines Bedrock Knowledge Base vector search with OpenSearch keyword search,
then merges results using Reciprocal Rank Fusion (RRF).

Why hybrid?
-----------
Pure vector (semantic) search is great for paraphrase recall — it surfaces
documents that *mean* the same thing even when worded differently.  But it
performs poorly on exact identifiers: product SKUs, error codes, version
numbers, proper nouns.  Keyword search handles those well.  RRF fusion
captures both signals without needing to tune score normalisation.

RRF formula
-----------
    score(d) = Σ  1 / (k + rank_i(d))

where k=60 (a constant that dampens the impact of very-high-ranked documents)
and the sum is over each ranker that returned document d.  Documents that
appear in both result sets receive a combined boost.

OpenSearch dependency
---------------------
The OpenSearch leg is optional.  If ``opensearch_endpoint`` is not supplied,
``HybridRetriever`` falls back to pure vector search.  The keyword search
implementation uses the ``opensearch-py`` package; install with::

    pip install bedrock-rag-patterns[opensearch]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from bedrock_rag.client import BedrockRAGClient

logger = logging.getLogger(__name__)

# RRF constant — 60 is the value used in the original paper (Cormack et al.)
# and works well in practice.  Lower values increase the impact of top ranks.
_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A single retrieved context chunk.

    Attributes
    ----------
    text:
        The raw text of the chunk.
    score:
        Merged RRF score (higher is better).
    source_uri:
        S3 URI or other source location reported by the Knowledge Base.
    metadata:
        Any additional metadata returned by the retriever.
    vector_rank:
        Rank in the vector search results (1-indexed), or None if not present.
    keyword_rank:
        Rank in the keyword search results (1-indexed), or None if not present.
    """

    text: str
    score: float
    source_uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_rank: int | None = None
    keyword_rank: int | None = None


class HybridRetriever:
    """Hybrid retriever combining Bedrock Knowledge Base and OpenSearch.

    Parameters
    ----------
    knowledge_base_id:
        Bedrock Knowledge Base ID.
    opensearch_endpoint:
        HTTPS endpoint of the OpenSearch / OpenSearch Serverless collection,
        e.g. ``https://abc123.us-east-1.aoss.amazonaws.com``.  Pass ``None``
        to disable keyword search.
    opensearch_index:
        Name of the OpenSearch index to search.  Required when
        ``opensearch_endpoint`` is set.
    opensearch_text_field:
        The field in the index that contains the document text.
    n_results:
        Number of results to fetch from *each* retriever before merging.
    rrf_k:
        RRF constant.  Defaults to 60.
    client:
        ``BedrockRAGClient`` instance.  A new one is created if not provided.
    region:
        AWS region, used only if ``client`` is not provided.
    """

    def __init__(
        self,
        knowledge_base_id: str,
        opensearch_endpoint: str | None = None,
        opensearch_index: str = "bedrock-knowledge-base-default-index",
        opensearch_text_field: str = "AMAZON_BEDROCK_TEXT_CHUNK",
        n_results: int = 10,
        rrf_k: int = _RRF_K,
        client: BedrockRAGClient | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.opensearch_endpoint = opensearch_endpoint
        self.opensearch_index = opensearch_index
        self.opensearch_text_field = opensearch_text_field
        self.n_results = n_results
        self.rrf_k = rrf_k
        self.client = client or BedrockRAGClient(region=region)

        self._os_client: Any = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        filter_expression: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and merge results for ``query``.

        Parameters
        ----------
        query:
            Natural-language query string.
        filter_expression:
            Optional Bedrock metadata filter applied to the vector search leg.

        Returns
        -------
        list[RetrievedChunk]
            Merged and RRF-ranked chunks, best first.
        """
        vector_results = self._vector_search(query, filter_expression)
        logger.debug("Vector search returned %d results", len(vector_results))

        if self.opensearch_endpoint:
            keyword_results = self._keyword_search(query)
            logger.debug("Keyword search returned %d results", len(keyword_results))
        else:
            keyword_results = []

        merged = self._rrf_merge(vector_results, keyword_results)
        logger.info(
            "HybridRetriever merged %d unique chunks (top score: %.4f)",
            len(merged),
            merged[0].score if merged else 0.0,
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        filter_expression: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Run vector search via Bedrock Knowledge Base Retrieve API."""
        raw = self.client.retrieve(
            knowledge_base_id=self.knowledge_base_id,
            query=query,
            n_results=self.n_results,
            filter_expression=filter_expression,
        )
        return raw

    def _keyword_search(self, query: str) -> list[dict[str, Any]]:
        """Run keyword search against OpenSearch using a multi-match query.

        Returns raw result dicts in the same shape as Bedrock Retrieve results
        so that ``_rrf_merge`` can handle both uniformly.

        Raises
        ------
        ImportError
            If ``opensearch-py`` is not installed.
        """
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection  # type: ignore[import]
            from requests_aws4auth import AWS4Auth  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "opensearch-py and requests-aws4auth are required for keyword search. "
                "Install with: pip install bedrock-rag-patterns[opensearch]"
            ) from exc

        if self._os_client is None:
            import boto3

            session = boto3.Session()
            credentials = session.get_credentials()
            region = self.client.region

            auth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                region,
                "aoss",
                session_token=credentials.token,
            )
            self._os_client = OpenSearch(
                hosts=[{"host": self.opensearch_endpoint, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )

        body = {
            "size": self.n_results,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [self.opensearch_text_field],
                    "type": "best_fields",
                }
            },
            "_source": [self.opensearch_text_field, "x-amz-bedrock-kb-source-uri"],
        }

        response = self._os_client.search(index=self.opensearch_index, body=body)
        hits = response.get("hits", {}).get("hits", [])

        # Normalise to Bedrock Retrieve result shape for uniform merging
        normalised = []
        for hit in hits:
            src = hit.get("_source", {})
            normalised.append(
                {
                    "content": {"text": src.get(self.opensearch_text_field, "")},
                    "location": {
                        "s3Location": {
                            "uri": src.get("x-amz-bedrock-kb-source-uri", "")
                        }
                    },
                    "metadata": {},
                    "_os_id": hit.get("_id", ""),
                }
            )
        return normalised

    def _rrf_merge(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[RetrievedChunk]:
        """Merge two ranked lists with Reciprocal Rank Fusion.

        Documents are identified by their text content (first 200 chars) to
        handle duplicates across the two result sets.  In practice the same
        chunk will appear verbatim in both OpenSearch and the Knowledge Base
        because both are backed by the same S3 source.
        """
        scores: dict[str, float] = {}
        vector_ranks: dict[str, int] = {}
        keyword_ranks: dict[str, int] = {}
        chunks_by_key: dict[str, dict[str, Any]] = {}

        def _key(raw: dict[str, Any]) -> str:
            text = raw.get("content", {}).get("text", "")
            return text[:200]  # fingerprint on first 200 chars

        for rank, raw in enumerate(vector_results, start=1):
            k = _key(raw)
            scores[k] = scores.get(k, 0.0) + 1.0 / (self.rrf_k + rank)
            vector_ranks[k] = rank
            chunks_by_key[k] = raw

        for rank, raw in enumerate(keyword_results, start=1):
            k = _key(raw)
            scores[k] = scores.get(k, 0.0) + 1.0 / (self.rrf_k + rank)
            keyword_ranks[k] = rank
            chunks_by_key.setdefault(k, raw)

        merged: list[RetrievedChunk] = []
        for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            raw = chunks_by_key[key]
            text = raw.get("content", {}).get("text", "")
            source_uri = (
                raw.get("location", {}).get("s3Location", {}).get("uri", "")
            )
            metadata = raw.get("metadata", {})

            merged.append(
                RetrievedChunk(
                    text=text,
                    score=score,
                    source_uri=source_uri,
                    metadata=metadata,
                    vector_rank=vector_ranks.get(key),
                    keyword_rank=keyword_ranks.get(key),
                )
            )

        return merged
