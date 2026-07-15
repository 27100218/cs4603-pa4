"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations

from functools import lru_cache

from databricks_langchain import DatabricksVectorSearch

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    s = get_settings()
    return DatabricksVectorSearch(
        endpoint=s["vs_endpoint"],
        index_name=s["vs_index"],
        columns=["chunk_to_retrieve", "source"],
    )


@lru_cache(maxsize=1)
def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
