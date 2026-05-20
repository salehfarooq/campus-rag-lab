from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def load_text(path: str | Path) -> list[Document]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    return [Document(text=text, metadata={"source": source.name, "kind": "text"})]


def load_pdf(path: str | Path) -> list[Document]:
    source = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to ingest PDF files.") from exc

    reader = PdfReader(str(source))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        docs.append(
            Document(
                text=page.extract_text() or "",
                metadata={"source": source.name, "kind": "pdf", "page": str(page_number)},
            )
        )
    return docs


def load_qa_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[Document] = []
    step = chunk_size - overlap
    for doc in documents:
        text = " ".join(doc.text.split())
        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size]
            if not chunk_text:
                continue
            metadata = {
                **doc.metadata,
                "chunk_start": str(start),
                "chunk_size": str(chunk_size),
                "overlap": str(overlap),
            }
            chunks.append(Document(text=chunk_text, metadata=metadata))
    return chunks

