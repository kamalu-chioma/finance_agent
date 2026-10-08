import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_agent.config import settings

logger = logging.getLogger(__name__)


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def _collection_path(ticker: str) -> str:
    return os.path.join(settings.chroma_persist_dir, ticker.upper())


def build_knowledge_base(
    ticker: str,
    news_items: list[dict],
    filing_docs: list[dict],
) -> int:
    """Chunk + embed news headlines and filing excerpts into a per-ticker Chroma collection.

    Returns the number of chunks written. Safe to call with empty inputs (writes nothing).
    """
    documents: list[Document] = []

    for item in news_items:
        text = f"{item['title']}. {item.get('summary', '')}".strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"type": "news", "source": item.get("publisher", "unknown"), "ticker": ticker.upper()},
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for filing in filing_docs:
        text = filing.get("text", "")
        if not text:
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "type": "filing",
                        "form": filing.get("form", "?"),
                        "filing_date": filing.get("filingDate", "?"),
                        "ticker": ticker.upper(),
                    },
                )
            )

    if not documents:
        logger.warning("No documents to ingest for %s", ticker)
        return 0

    Chroma.from_documents(
        documents,
        embedding=_embeddings(),
        persist_directory=_collection_path(ticker),
        collection_name=f"ticker_{ticker.lower()}",
    )
    return len(documents)
