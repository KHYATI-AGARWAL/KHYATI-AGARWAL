from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "what",
    "which",
    "how",
}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str


@dataclass
class RetrievalHit:
    chunk_id: str
    source: str
    title: str
    text: str
    score: float


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def split_into_chunks(text: str, chunk_size: int = 110, overlap: int = 25) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.doc_tokens = [tokenize(c.text) for c in self.chunks]

        doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0

        freq: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                freq[token] = freq.get(token, 0) + 1
        self.doc_freq = freq

    def _idf(self, token: str) -> float:
        n = len(self.doc_tokens)
        df = self.doc_freq.get(token, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalHit]:
        q_tokens = tokenize(query_text)
        if not q_tokens or not self.chunks:
            return []

        hits: List[RetrievalHit] = []
        for idx, tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = len(tokens)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            for token in q_tokens:
                f = tf.get(token, 0)
                if f == 0:
                    continue
                idf = self._idf(token)
                denom = f + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1)))
                score += idf * ((f * (self.k1 + 1)) / denom)

            if score > 0:
                c = self.chunks[idx]
                hits.append(
                    RetrievalHit(
                        chunk_id=c.chunk_id,
                        source=c.source,
                        title=c.title,
                        text=c.text,
                        score=round(score, 4),
                    )
                )

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]


def load_text_documents(corpus_dir: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        docs.append((path.name, path.read_text(encoding="utf-8").strip()))
    return [(name, text) for name, text in docs if text]


def build_chunks(corpus_dir: Path, chunk_size: int = 110, overlap: int = 25) -> List[Chunk]:
    chunks: List[Chunk] = []
    for filename, text in load_text_documents(corpus_dir):
        title = filename.replace("_", " ").replace(".txt", "").title()
        split_chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text in enumerate(split_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}::chunk-{i + 1}",
                    source=filename,
                    title=title,
                    text=chunk_text,
                )
            )
    return chunks


class AyurvedicGastroRAG:
    def __init__(self, index: BM25Index):
        self.index = index

    def answer(self, question: str, top_k: int = 4) -> str:
        hits = self.index.query(question, top_k=top_k)
        if not hits:
            return (
                "I could not retrieve enough gastroenterology-focused Ayurvedic context. "
                "Please expand the corpus with authoritative sources."
            )

        evidence_lines = []
        citations = []
        for i, hit in enumerate(hits, start=1):
            evidence_lines.append(f"[{i}] ({hit.source}) {hit.text}")
            citations.append(f"[{i}] {hit.source}:{hit.chunk_id}")

        response = (
            "Ayurvedic Gastro Assistant (RAG Draft)\n"
            "-----------------------------------\n"
            f"Question: {question}\n\n"
            "Evidence retrieved:\n"
            + "\n".join(evidence_lines)
            + "\n\n"
            "Draft answer for your paper:\n"
            "Based on the retrieved Ayurvedic gastroenterology context, the response should link "
            "the symptom pattern to relevant dosha imbalance (for example, agni dysregulation, "
            "ama accumulation, or vata-pitta aggravation), then map this to dietary protocol, "
            "lifestyle intervention, and classical formulations described in the corpus. "
            "For publication quality, pair this knowledge-grounded narrative with modern clinical "
            "endpoints (symptom score, stool frequency, pain score, inflammatory markers) and "
            "clearly separate evidence-backed claims from hypotheses.\n\n"
            "Safety note: this assistant is for research drafting and not a substitute for clinical "
            "diagnosis, emergency care, or personalized prescribing.\n\n"
            "Citations:\n"
            + "\n".join(citations)
        )
        return response


def save_chunks(chunks: Sequence[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")


def load_chunks(path: Path) -> List[Chunk]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**record) for record in records]


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Ayurvedic Gastro RAG assistant")
    parser.add_argument("--corpus_dir", default="data/ayurveda_gastro_knowledge")
    parser.add_argument("--index_file", default="data/index/ayurvedic_gastro_chunks.json")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--question", help="Question to ask the RAG assistant")
    parser.add_argument("--top_k", type=int, default=4)
    args = parser.parse_args()

    index_file = Path(args.index_file)

    if args.build_index:
        chunks = build_chunks(Path(args.corpus_dir))
        save_chunks(chunks, index_file)
        print(f"Index built with {len(chunks)} chunks at {index_file}")
        return

    if not args.question:
        raise ValueError("Provide --question when not using --build_index")

    if not index_file.exists():
        chunks = build_chunks(Path(args.corpus_dir))
        save_chunks(chunks, index_file)
    else:
        chunks = load_chunks(index_file)

    bm25 = BM25Index()
    bm25.build(chunks)
    assistant = AyurvedicGastroRAG(bm25)
    print(assistant.answer(args.question, top_k=args.top_k))


if __name__ == "__main__":
    run_cli()
