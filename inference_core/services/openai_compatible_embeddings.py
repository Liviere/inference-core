"""Embeddings from a server that speaks OpenAI's ``/embeddings``.

LangChain's ``OpenAIEmbeddings`` is made for OpenAI itself: it tokenizes the
texts with tiktoken and sends token ids unless told not to, and it throws the
response's ``usage`` away. A gateway or a hosted model behind an OpenAI-shaped
endpoint wants the texts, and the usage is what the call is priced from. This
client sends the texts as they are, asks for floats, adds the provider's
``request_options`` to the body and returns the usage with the vectors.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference_core.services.embedding_usage import ReportedUsage

DEFAULT_BATCH_SIZE = 64
DEFAULT_TIMEOUT_SECONDS = 60.0


class OpenAICompatibleEmbeddings:
    """A LangChain-style embeddings client that also reports usage."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: Optional[str],
        request_options: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not base_url:
            raise ValueError(f"Embedding model '{model}' has no base_url")
        if not api_key:
            raise ValueError(f"Embedding model '{model}' has no API key")
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._extra_body = dict(request_options or {})
        self._batch_size = batch_size or DEFAULT_BATCH_SIZE
        self._timeout = timeout
        self._client: Any = None
        self._async_client: Any = None

    # -- LangChain's Embeddings interface ---------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_with_usage(texts)[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return (await self.aembed_with_usage(texts))[0]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]

    # -- With usage --------------------------------------------------------

    def embed_with_usage(
        self, texts: Sequence[str]
    ) -> Tuple[List[List[float]], ReportedUsage]:
        vectors: List[List[float]] = []
        usage = _Usage()
        for batch in self._batches(texts):
            response = self._sync_client().embeddings.create(
                **self._request(batch)
            )
            vectors.extend(_vectors(response, len(batch)))
            usage.add(response)
        return vectors, usage.reported()

    async def aembed_with_usage(
        self, texts: Sequence[str]
    ) -> Tuple[List[List[float]], ReportedUsage]:
        vectors: List[List[float]] = []
        usage = _Usage()
        for batch in self._batches(texts):
            response = await self._async_client_().embeddings.create(
                **self._request(batch)
            )
            vectors.extend(_vectors(response, len(batch)))
            usage.add(response)
        return vectors, usage.reported()

    # -- Internals -----------------------------------------------------------

    def _request(self, batch: List[str]) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        if self._extra_body:
            request["extra_body"] = dict(self._extra_body)
        return request

    def _batches(self, texts: Sequence[str]):
        items = list(texts)
        for start in range(0, len(items), self._batch_size):
            yield items[start : start + self._batch_size]

    def _sync_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key, base_url=self.base_url, timeout=self._timeout
            )
        return self._client

    def _async_client_(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self._api_key, base_url=self.base_url, timeout=self._timeout
            )
        return self._async_client


def _vectors(response: Any, expected: int) -> List[List[float]]:
    data = sorted(response.data, key=lambda item: item.index)
    if len(data) != expected:
        raise ValueError(
            f"Embedding response has {len(data)} vectors for {expected} texts"
        )
    return [list(item.embedding) for item in data]


class _Usage:
    """The usage of a call's batches, added up."""

    def __init__(self) -> None:
        self.tokens: Optional[int] = None
        self.cost: Optional[float] = None
        self.model: Optional[str] = None

    def add(self, response: Any) -> None:
        self.model = self.model or _text(getattr(response, "model", None))
        usage = getattr(response, "usage", None)
        tokens = _number(getattr(usage, "prompt_tokens", None))
        if tokens is None:
            tokens = _number(getattr(usage, "total_tokens", None))
        if tokens is not None:
            self.tokens = int((self.tokens or 0) + tokens)
        cost = _number(getattr(usage, "cost", None))
        if cost is None:
            extra = getattr(usage, "model_extra", None) or {}
            cost = _number(extra.get("cost")) if isinstance(extra, dict) else None
        if cost is not None:
            self.cost = (self.cost or 0.0) + cost

    def reported(self) -> ReportedUsage:
        return ReportedUsage(
            input_tokens=self.tokens, reported_model=self.model, provider_cost=self.cost
        )


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None
