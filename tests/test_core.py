from campus_rag_lab.cache import SemanticResponseCache
from campus_rag_lab.config import load_settings
from campus_rag_lab.generation import CampusRagService, GeminiGroundedGenerator
from campus_rag_lab.ingestion import Document, chunk_documents
from campus_rag_lab.retrieval import LocalVectorStore


def test_chunk_documents_validates_overlap():
    docs = [Document("abcdef")]
    chunks = chunk_documents(docs, chunk_size=3, overlap=1)
    assert [chunk.text for chunk in chunks] == ["abc", "cde", "ef"]


def test_local_retrieval_finds_relevant_chunk():
    store = LocalVectorStore()
    store.add_documents([Document("Attendance requires 80 percent classes.")])
    results = store.search("attendance requirement", top_k=1, threshold=0.0)
    assert results
    assert "Attendance" in results[0].text


def test_semantic_cache_hits_similar_question():
    cache = SemanticResponseCache(threshold=0.3)
    cache.store("What is the attendance requirement?", "80 percent")
    lookup = cache.lookup("attendance requirement?")
    assert lookup.hit
    assert lookup.response == "80 percent"


def test_offline_service_abstains_without_context():
    settings = load_settings()
    store = LocalVectorStore()
    service = CampusRagService(store, GeminiGroundedGenerator(settings))
    trace = service.ask("What is the policy?")
    assert trace.answer == "I don't know"
    assert not trace.answerable

