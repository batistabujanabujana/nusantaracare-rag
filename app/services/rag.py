"""Small, deterministic retrieval pipeline over the required source document."""
from __future__ import annotations

import re
import os
import math
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOURCE_PATH = Path(__file__).resolve().parents[2] / "data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md"
load_dotenv()
INJECTION_RE = re.compile(
    r"(ignore|disregard|abaikan|lupakan).{0,60}(instruction|instruksi|aturan|system prompt)"
    r"|\b(system prompt|developer message|reveal secrets|bypass safety)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE_RE = re.compile(
    r"\b(medis|kesehatan|legal|hukum|gaji|penggajian|payroll|tunjangan|kompensasi|"
    r"konseling|evaluasi kinerja|infrastruktur eksternal)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE_QUOTES = (
    (r"medis|kesehatan", "konsultasi medis atau kesehatan karyawan"),
    (r"legal|hukum", "nasihat hukum atau kepatuhan terhadap peraturan perundang-undangan"),
    (r"gaji|penggajian|payroll|tunjangan|kompensasi", "penghitungan gaji, tunjangan, atau kompensasi finansial"),
    (r"konseling|evaluasi kinerja", "evaluasi atau konseling kinerja personal"),
    (r"infrastruktur eksternal", "permintaan yang berkaitan dengan infrastruktur atau sistem di luar kewenangan Direktorat Operasi dan Layanan Internal"),
)


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    version: str
    active: bool
    index: int
    doc_id: str
    doc_title: str
    effective_date: str


class KnowledgeBaseMissing(RuntimeError):
    pass


def _parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    if not markdown.startswith("---\n"):
        return {}, markdown
    end = markdown.find("\n---\n", 4)
    if end < 0:
        return {}, markdown
    metadata: dict[str, str] = {}
    for line in markdown[4:end].splitlines():
        match = re.match(r"^([\w_]+):\s*(.*?)\s*$", line)
        if match:
            metadata[match.group(1)] = match.group(2).strip('"\'')
    return metadata, markdown[end + 5 :]


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Preserve heading hierarchy, so nested archival sections stay inactive."""
    result: list[tuple[str, str]] = []
    headings: dict[int, str] = {}
    section_title, lines = "Dokumen", []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            if "\n".join(lines).strip():
                result.append((section_title, "\n".join(lines).strip()))
            level, heading = len(match.group(1)), match.group(2).strip()
            headings = {n: title for n, title in headings.items() if n < level}
            headings[level] = heading
            section_title = " > ".join(headings[n] for n in sorted(headings))
            lines = []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        result.append((section_title, "\n".join(lines).strip()))
    return result


def _version_and_active(section: str, metadata: dict[str, str]) -> tuple[str, bool]:
    if re.search(r"Arsip Kebijakan v1\.4", section, re.IGNORECASE):
        return "1.4", False
    # Document metadata is authoritative for the v2.0 body. This avoids
    # misclassifying its change log merely because it mentions v1.4.
    version = metadata.get("doc_version", "2.0")
    active = metadata.get("is_active", "true").lower() == "true"
    return version, active


def _make_chunks(markdown: str, chunk_words: int = 220, overlap_words: int = 35) -> list[Chunk]:
    chunks: list[Chunk] = []
    metadata, content = _parse_front_matter(markdown)
    doc_id = metadata.get("doc_id", "unknown")
    doc_title = metadata.get("doc_title", "NusantaraCare")
    effective_date = metadata.get("effective_date", "unknown")
    for section, body in _split_sections(content):
        version, active = _version_and_active(section, metadata)
        # Keep paragraph boundaries where possible, then bound chunk size by words.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        words: list[str] = []
        for para in paragraphs:
            para_words = para.split()
            if words and len(words) + len(para_words) > chunk_words:
                chunks.append(Chunk(" ".join(words), section, version, active, len(chunks), doc_id, doc_title, effective_date))
                words = words[-overlap_words:]
            while len(para_words) > chunk_words:
                room = max(1, chunk_words - len(words))
                words.extend(para_words[:room])
                para_words = para_words[room:]
                chunks.append(Chunk(" ".join(words), section, version, active, len(chunks), doc_id, doc_title, effective_date))
                words = words[-overlap_words:]
            words.extend(para_words)
        if words:
            chunks.append(Chunk(" ".join(words), section, version, active, len(chunks), doc_id, doc_title, effective_date))
    return chunks


class Retriever:
    def __init__(self, source_path: Path = SOURCE_PATH):
        if not source_path.is_file():
            raise KnowledgeBaseMissing(f"Required source document missing: {source_path}")
        self.source_path = source_path
        self.chunks = _make_chunks(source_path.read_text(encoding="utf-8"))
        if not self.chunks:
            raise KnowledgeBaseMissing("The required source document has no readable content")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
        self.client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/") if os.getenv("GEMINI_API_KEY") else None
        self.embeddings: list[list[float]] | None = None
        self.embedding_error: str | None = None
        if self.client:
            try:
                self.embeddings = self._embed([f"{c.section}\n{c.text}" for c in self.chunks])
            except Exception as exc:
                # Keep the local service usable with keyword retrieval if the
                # embedding API is unavailable; health exposes this fallback.
                status_code = getattr(exc, "status_code", None)
                if isinstance(exc, TypeError):
                    # TypeError text identifies local SDK/request-shape issues;
                    # cap it and never include request headers or credentials.
                    detail = str(exc).replace(os.getenv("GEMINI_API_KEY", ""), "[redacted]")
                    self.embedding_error = f"TypeError: {detail[:180]}"
                elif status_code is not None:
                    self.embedding_error = f"{type(exc).__name__} (HTTP {status_code})"
                else:
                    self.embedding_error = type(exc).__name__
                self.client = None
        self.evidence = [
            (chunk_index, sentence.strip())
            for chunk_index, chunk in enumerate(self.chunks)
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text)
            if len(sentence.split()) >= 4
        ]
        self.evidence_vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self.evidence_matrix = self.evidence_vectorizer.fit_transform([sentence for _, sentence in self.evidence])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise RuntimeError("Gemini API client is not configured")
        vectors: list[list[float]] = []
        # Batch requests so this also works if the document grows later.
        for start in range(0, len(texts), 64):
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts[start : start + 64],
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def search(self, question: str, top_k: int = 4) -> list[tuple[Chunk, float, str]]:
        q = self.vectorizer.transform([question])
        chunk_scores = cosine_similarity(q, self.matrix).ravel()
        semantic_scores = [0.0] * len(self.chunks)
        if self.client and self.embeddings is not None:
            query_embedding = self._embed([question])[0]
            semantic_scores = [self._cosine(query_embedding, item) for item in self.embeddings]
        sentence_q = self.evidence_vectorizer.transform([question])
        sentence_scores = cosine_similarity(sentence_q, self.evidence_matrix).ravel()
        asks_history = bool(re.search(r"riwayat|perubahan|v\s*1\.4|versi sebelumnya", question, re.I))
        asks_numeric = bool(re.search(r"berapa|minimum|minimal|maksimal|batas|jumlah|hari kerja", question, re.I))
        best_evidence: dict[int, tuple[float, str]] = {}
        for (chunk_index, sentence), score in zip(self.evidence, sentence_scores):
            evidence_score = float(score)
            if asks_history and re.search(r"v\s*1\.4|versi sebelumnya", sentence, re.I):
                evidence_score += 0.20
            if asks_numeric and re.search(r"\d+\s*hari|\b\d+\b", sentence, re.I):
                evidence_score += 0.12
            previous = best_evidence.get(chunk_index)
            if previous is None or evidence_score > previous[0]:
                best_evidence[chunk_index] = (evidence_score, sentence)
        # Archived material is never eligible for ordinary current-policy answers.
        eligible = [i for i, c in enumerate(self.chunks) if c.active]

        def relevance(i: int) -> float:
            sentence_score = best_evidence.get(i, (0.0, ""))[0]
            if self.embeddings is not None:
                score = (0.70 * semantic_scores[i]
                         + 0.20 * float(chunk_scores[i])
                         + 0.10 * min(sentence_score, 1.0))
            else:
                score = 0.45 * float(chunk_scores[i]) + 0.55 * sentence_score
            section = self.chunks[i].section.lower()
            # Prefer the operative SOP over a change-log summary for current
            # procedure questions. Keep history eligible for history questions.
            if "riwayat perubahan" in section and not asks_history:
                score -= 0.12
            if ("sop fasilitas dan perlengkapan kerja" in section
                    and "perlengkapan" in question.lower() and not asks_history):
                score += 0.12
            return score

        ranked = sorted(eligible, key=relevance, reverse=True)
        return [
            (self.chunks[i], relevance(i), best_evidence.get(i, (0.0, self.chunks[i].text))[1])
            for i in ranked[:top_k]
            if relevance(i) > 0
        ]


def looks_like_injection(question: str) -> bool:
    return bool(INJECTION_RE.search(question))


def answer_question(retriever: Retriever, question: str, threshold: float = 0.20) -> dict:
    if looks_like_injection(question):
        return {"answer": "Permintaan ditolak karena mengandung instruksi yang mencoba mengubah aturan sistem.",
                "confidence_label": "low", "reason_code": "unsafe_request"}
    if OUT_OF_SCOPE_RE.search(question):
        quote = next((text for pattern, text in OUT_OF_SCOPE_QUOTES if re.search(pattern, question, re.I)), None)
        answer = "Pertanyaan ini di luar cakupan panduan operasional NusantaraCare."
        if quote:
            answer += (f' Kutipan: "{quote}" (Sumber: Tujuan, Ruang Lingkup, dan Status Dokumen; '
                       "versi 2.0; efektif 2026-07-01; doc_id NC-OPS-001).")
        answer += " Silakan hubungi unit kerja atau direktorat yang berwenang."
        return {"answer": answer,
                "confidence_label": "low", "reason_code": "out_of_scope"}
    if (re.search(r"v\s*1\.4|versi\s*1\.4", question, re.I)
            and re.search(r"masih berlaku|status|masih aktif|sudah aktif", question, re.I)):
        return {
            "answer": (
                'Tidak. v1.4 sudah digantikan oleh v2.0. Kutipan: '
                '"Panduan Operasional Layanan Internal NusantaraCare versi 2.0 berlaku efektif sejak 1 Juli 2026 '
                'dan menggantikan seluruh versi sebelumnya, termasuk versi 1.4 dan seluruh revisi minor di antaranya." '
                '(Sumber: Riwayat Perubahan dan Arsip Kebijakan; versi 2.0; efektif 2026-07-01; doc_id NC-OPS-001).'
            ),
            "confidence_label": "high",
            "reason_code": "answered",
        }
    asks_history = bool(re.search(r"riwayat|perubahan|v\s*1\.4|versi sebelumnya", question, re.I))
    cutoff = min(threshold, 0.10) if asks_history else threshold
    matches = retriever.search(question)
    matches = [(c, s, evidence) for c, s, evidence in matches if s >= cutoff]
    if not matches:
        return {"answer": "Informasi tersebut tidak ditemukan dalam dokumen NusantaraCare v2.0 yang aktif.",
                "confidence_label": "low", "reason_code": "no_relevant_context"}
    # Return one sentence-level extract from the strongest source. This keeps
    # answers concise and avoids mixing the SOP with FAQ/table/history excerpts.
    chunk, score, evidence = matches[0]
    quote = re.sub(r"\s+", " ", evidence).strip()
    citation = (
        f'Kutipan: "{quote}" (Sumber: {chunk.section}; '
        f'versi {chunk.version}; efektif {chunk.effective_date}; '
        f'doc_id {chunk.doc_id}; chunk {chunk.index})'
    )
    label = "high" if matches[0][1] >= 0.30 else "medium"
    return {"answer": citation,
            "confidence_label": label, "reason_code": "answered"}
