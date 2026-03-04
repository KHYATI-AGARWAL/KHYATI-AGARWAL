from pathlib import Path

from src.ayurvedic_gastro_rag import BM25Index, AyurvedicGastroRAG, build_chunks, split_into_chunks


def test_split_into_chunks_produces_multiple_chunks_for_long_text():
    text = "word " * 280
    chunks = split_into_chunks(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 3


def test_retrieval_hits_relevant_gastro_context():
    chunks = build_chunks(Path("data/ayurveda_gastro_knowledge"), chunk_size=80, overlap=10)
    index = BM25Index()
    index.build(chunks)

    hits = index.query("What are alarm symptoms and safety in GI cases?", top_k=3)
    assert hits
    assert any("safety" in hit.text.lower() or "alarm" in hit.text.lower() for hit in hits)


def test_answer_contains_citations_block():
    chunks = build_chunks(Path("data/ayurveda_gastro_knowledge"), chunk_size=80, overlap=10)
    index = BM25Index()
    index.build(chunks)
    assistant = AyurvedicGastroRAG(index)

    response = assistant.answer("How should I report interventions for IBS in Ayurveda?", top_k=3)
    assert "Citations:" in response
    assert "[1]" in response
