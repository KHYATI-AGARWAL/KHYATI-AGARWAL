# Ayurvedic Project: Gastroenterology-Focused RAG Assistant

This branch (`ayurvedic-project`) contains a **full retrieval-augmented generation (RAG) baseline** for your paper topic:

> *A large language model assistant for gastroenterology in traditional Ayurvedic medicine.*

The implementation is designed for **research writing workflows**:
- retrieve evidence snippets from a curated Ayurvedic gastro corpus,
- ground responses with explicit source citations,
- produce paper-ready draft text with safety disclaimers.

## What is included

- `src/ayurvedic_gastro_rag.py`
  - document loading and chunking,
  - lightweight BM25 retriever (pure Python, no external vector DB required),
  - response synthesizer with citations,
  - CLI for indexing and querying.
- `data/ayurveda_gastro_knowledge/*.txt`
  - starter domain corpus focused on agni/ama, functional GI patterns, interventions and safety.
- `tests/test_ayurvedic_gastro_rag.py`
  - unit tests for chunking, retrieval relevance, and response citation format.

## Quickstart

```bash
# Build the chunk index
python src/ayurvedic_gastro_rag.py --build_index

# Ask a gastro-focused question
python src/ayurvedic_gastro_rag.py \
  --question "How should I frame an Ayurvedic interpretation of IBS-like symptoms in my paper?"
```

## Example research questions

- "How does agni imbalance relate to functional dyspepsia?"
- "What safety language should be included in an Ayurvedic gastro assistant paper?"
- "How can I structure multimodal intervention reporting (diet, lifestyle, herbal)?"

## Suggested paper method section wording

You can describe the system as:
1. **Corpus construction**: Curated gastroenterology-relevant Ayurvedic text snippets.
2. **Segmentation**: Fixed-size chunking with overlap to preserve context continuity.
3. **Retrieval**: BM25 lexical relevance scoring over chunks.
4. **Grounded generation**: Prompted narrative constrained by top-k retrieved evidence.
5. **Safety layer**: Mandatory non-diagnostic disclaimer and red-flag escalation language.

## Next upgrades (for publication depth)

- Swap BM25 with embedding retrieval (e.g., bge/e5) and reranker.
- Add structured patient profile input (symptoms, tongue/stool patterns, triggers).
- Add dual-output mode:
  - *Ayurvedic interpretation* and
  - *Biomedical differential red-flag checklist*.
- Add evaluation metrics:
  - retrieval precision@k,
  - citation faithfulness,
  - hallucination rate,
  - expert rubric score (Ayurveda clinician + GI specialist).

## Safety note

This codebase supports academic writing and prototyping. It is **not** a clinical decision system and should not be used for direct diagnosis or emergency triage.
