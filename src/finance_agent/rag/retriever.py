import logging
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from finance_agent.rag.ingest import _collection_path, _embeddings

logger = logging.getLogger(__name__)


def retrieve_context(ticker: str, query: str, k: int = 6) -> list[Document]:
    """Similarity search over the ticker's knowledge base. Returns [] if none was built."""
    path = _collection_path(ticker)
    if not os.path.isdir(path):
        return []
    try:
        store = Chroma(
            persist_directory=path,
            embedding_function=_embeddings(),
            collection_name=f"ticker_{ticker.lower()}",
        )
        return store.similarity_search(query, k=k)
    except Exception:
        logger.exception("Retrieval failed for %s", ticker)
        return []
