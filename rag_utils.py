"""
Core RAG (Retrieval-Augmented Generation) logic.

Kept deliberately unabstracted (no LangChain) so every step is visible —
this is the part you should be able to explain line-by-line in an interview.
"""

import os
import time
import numpy as np
import faiss
from pypdf import PdfReader
from google import genai
from google.genai import errors
from google.genai.types import EmbedContentConfig

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-flash-latest"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # overlap so we don't cut a relevant sentence in half
TOP_K = 3             # how many chunks to retrieve per question

EMBED_BATCH_SIZE = 100          # the embedding API accepts at most 100 texts per call
MAX_CHUNKS = 400                # safety cap so one document can't burn the whole free-tier quota
RATE_LIMIT_WAIT_SECONDS = 60    # how long to wait when the free tier says "slow down" (HTTP 429)
MAX_RETRIES = 4


def get_client() -> genai.Client:
    """Create a Gemini client from the GEMINI_API_KEY environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def get_page_count(file) -> int:
    """How many pages does this PDF have?"""
    file.seek(0)
    return len(PdfReader(file).pages)


def extract_text_from_pdf(file, start_page: int = 1, end_page: int | None = None) -> str:
    """
    Pull raw text out of a PDF, only for pages start_page..end_page
    (1-indexed, inclusive). Indexing a whole textbook is slow and can exceed
    the free-tier quota, so we let the user pick a range.
    """
    file.seek(0)
    reader = PdfReader(file)
    pages = reader.pages[start_page - 1:end_page]
    text = ""
    for page in pages:
        text += (page.extract_text() or "") + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of roughly chunk_size characters.

    - Whitespace is normalised first (PDF text is full of hard line breaks).
    - Chunks end on a word boundary, so no word is cut in half.
    - Overlap matters: without it, a sentence that straddles two chunks
      can lose meaning in both halves.
    """
    text = " ".join(text.split())
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def _embed_with_retry(client: genai.Client, texts: list[str], task_type: str, on_wait=None):
    """
    Call the embedding API. On the free tier you can hit a per-minute limit
    (HTTP 429); instead of crashing we wait a minute and try again.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=EmbedContentConfig(task_type=task_type),
            )
        except errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                if on_wait:
                    on_wait(RATE_LIMIT_WAIT_SECONDS)
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            raise


def embed_chunks(client: genai.Client, chunks: list[str], progress_callback=None) -> np.ndarray:
    """
    Embed document chunks for storage, in batches of EMBED_BATCH_SIZE.

    The API rejects a request with more than 100 texts (that was the
    "at most 100 requests can be in one batch" error), so we loop.

    task_type=RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY (used in embed_query below)
    is asymmetric retrieval — the model embeds a question and its answer
    slightly differently so that matching them works better.

    progress_callback(done, total, message) lets the UI show a progress bar.
    """
    total = len(chunks)
    all_vectors = []

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]

        def on_wait(seconds, done=len(all_vectors)):
            if progress_callback:
                progress_callback(done, total, f"Free-tier rate limit hit — waiting {seconds}s, then retrying...")

        response = _embed_with_retry(client, batch, "RETRIEVAL_DOCUMENT", on_wait=on_wait)

        # Safety check: vectors must line up 1:1 with chunks, otherwise
        # retrieval would silently return the wrong text.
        if len(response.embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(response.embeddings)} vectors for {len(batch)} chunks."
            )

        all_vectors.extend(e.values for e in response.embeddings)
        if progress_callback:
            progress_callback(len(all_vectors), total, None)

    return np.array(all_vectors, dtype="float32")


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    """Embed a single user question, tagged as a query (not a document)."""
    response = _embed_with_retry(client, [query], "RETRIEVAL_QUERY")
    return np.array([response.embeddings[0].values], dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build an in-memory FAISS index over the chunk vectors.

    IndexFlatL2 checks every vector (exact search) — fine at this scale.
    Production systems with millions of vectors use approximate indexes
    (IVF, HNSW) that trade a little accuracy for a lot of speed.
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


def retrieve_top_chunks(
    index: faiss.IndexFlatL2, query_vector: np.ndarray, chunks: list[str], k: int = TOP_K
) -> list[str]:
    """Find the k chunks whose embeddings are closest to the query embedding."""
    _distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def generate_answer(client: genai.Client, question: str, retrieved_chunks: list[str]) -> str:
    """
    Ask Gemini to answer using ONLY the retrieved context.

    This instruction is what keeps answers grounded instead of the model
    falling back on its own possibly-outdated or irrelevant training data.
    """
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know — do not guess.

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text