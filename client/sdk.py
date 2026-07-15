"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._endpoint = endpoint_name
        self._host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self._token = token or os.environ.get("DATABRICKS_TOKEN", "")
        self._timeout = timeout
        self._max_retries = max_retries

        if not self._host or not self._token:
            raise ValueError(
                "DATABRICKS_HOST and DATABRICKS_TOKEN must be provided or set in environment."
            )

    def _client(self):
        import openai
        return openai.OpenAI(
            api_key=self._token,
            base_url=f"{self._host}/serving-endpoints",
            timeout=self._timeout,
            max_retries=0,
        )

    def health_check(self) -> bool:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(host=self._host, token=self._token)
        try:
            ep = w.serving_endpoints.get(self._endpoint)
            ready = getattr(ep.state, "ready", None)
            # The SDK may return an enum (ServingEndpointStateReady.READY) or a
            # plain string.  Use .value if present, else fall back to str().
            ready_str = getattr(ready, "value", str(ready)).upper()
            return ready_str == "READY"
        except Exception:
            return False

    def ask(self, question: str) -> str:
        import openai
        client = self._client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self._endpoint,
                    messages=[{"role": "user", "content": question}],
                )
                return resp.choices[0].message.content or ""
            except openai.RateLimitError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
            except openai.APIStatusError as exc:
                if exc.status_code == 503:
                    last_exc = exc
                    time.sleep(2 ** attempt)
                else:
                    raise AnalystClientError(
                        str(exc.message), exc.status_code, exc.request_id or ""
                    ) from exc
            except openai.APITimeoutError as exc:
                raise TimeoutError(f"Request timed out after {self._timeout}s") from exc

        raise last_exc  # type: ignore[misc]

    def ask_streaming(self, question: str) -> Iterator[str]:
        import openai
        client = self._client()

        # LangGraph runs to completion before returning, so the endpoint cannot
        # emit token-by-token chunks.  Use a blocking call and yield the full
        # content so callers can use the iterator interface unchanged.
        try:
            resp = client.chat.completions.create(
                model=self._endpoint,
                messages=[{"role": "user", "content": question}],
            )
            content = resp.choices[0].message.content or ""
            yield content
        except openai.APITimeoutError as exc:
            raise TimeoutError(f"Stream timed out after {self._timeout}s") from exc
        except openai.APIStatusError as exc:
            raise AnalystClientError(
                str(exc.message), exc.status_code, exc.request_id or ""
            ) from exc
