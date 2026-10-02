# Chat with your notes — a RAG project

Upload a PDF, then ask questions about it. The app retrieves the most relevant
parts of your document and asks Gemini to answer using only that retrieved
text — instead of guessing from its own general training data.

## Setup

1. **Get a free Gemini API key** — go to https://aistudio.google.com/app/apikey,
   sign in with any Google account, and create a key. No credit card needed.
2. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
4. **Add your API key**: copy `.env.example` to a new file named `.env`, and
   paste your key in place of `your_api_key_here`.
5. **Run it**:
   ```
   streamlit run app.py
   ```
   This opens the app in your browser at `localhost:8501`.

## Theme (important for the look)

The colours come from `.streamlit/config.toml`, which must sit in the project
folder next to `app.py` (so `streamlit run app.py` finds it). Without it the app
still works but uses Streamlit's default colours, and a small note appears in
the sidebar. The reading font (Literata) loads from Google Fonts, so it needs
an internet connection; offline it falls back to a system serif font.

## Using it (and the free-tier limits)

Upload a PDF, choose a **page range** (start with one chapter, roughly 20–50
pages), and click *Process document*. Then ask questions.

Why the page range? The free Gemini tier limits embedding calls — roughly
100 requests/min, 30,000 tokens/min and 1,000 requests/day for
`gemini-embedding-001` (Google changes these, so check AI Studio for your
project's real numbers). A whole textbook can be thousands of chunks, more
than a day's quota. So the app:

- sends chunks in **batches of 100** (the API's hard maximum per call),
- **waits and retries** automatically if it hits the per-minute limit (you'll
  see a message on the progress bar),
- refuses to index more than `MAX_CHUNKS` (400) chunks at once — raise this
  constant in `rag_utils.py` if you move to a paid tier.

## How it works

Two phases, matching the two diagrams from our chat:

**Indexing** (`rag_utils.py`, runs once per document): the PDF's text is
pulled out, split into overlapping chunks (`chunk_text`), each chunk is
turned into a numeric embedding (`embed_chunks`), and all the embeddings go
into a FAISS index (`build_index`) — a searchable store of vectors.

**Query** (runs every time you ask a question): your question is embedded
too (`embed_query`), FAISS finds the chunks whose embeddings are closest to
it (`retrieve_top_chunks`), and those chunks plus your question are sent to
Gemini (`generate_answer`), which is told explicitly to answer only from
that context.

## Things to be ready to explain in an interview

These are the design decisions an interviewer is most likely to poke at —
worth being able to answer in your own words, not just recite:

- **Why chunk at all, instead of sending the whole document?** Embedding
  models and LLM context windows both have limits, and retrieval is more
  precise over small, focused pieces than one giant blob of text.
- **Why overlap the chunks?** Without overlap, a sentence that falls right
  on a chunk boundary gets cut in half and loses meaning in both pieces.
- **Why does `embed_chunks` use `RETRIEVAL_DOCUMENT` and `embed_query` use
  `RETRIEVAL_QUERY`?** This is asymmetric retrieval — questions and answers
  are phrased differently, so the model embeds them with different hints to
  make matching between the two more accurate.
- **Why FAISS instead of a plain Python loop comparing vectors?** At this
  scale (a few hundred chunks) it barely matters — but FAISS is what
  production systems use once you have millions of vectors, because brute
  force gets too slow. `IndexFlatL2` here is still exact search; FAISS also
  offers approximate indexes (IVF, HNSW) that trade a little accuracy for
  a lot of speed at scale.
- **What stops the model from making things up?** The prompt in
  `generate_answer` explicitly instructs it to answer only from the given
  context and say "I don't know" otherwise. It's not foolproof, but it
  substantially reduces hallucination compared to an unconstrained prompt.
- **How did you deal with API limits?** The embedding API accepts at most 100
  texts per call, and the free tier has per-minute and per-day quotas. So
  `embed_chunks` batches the work, `_embed_with_retry` waits and retries on
  HTTP 429, and the app caps chunks per document. (This came from a real bug
  you hit and fixed — a good story to tell.)

## Ideas to extend (do this after the base version works)

- **Evaluate it** — this is the part almost nobody does, which is exactly
  why it stands out. Write ~10 question/answer pairs you know the true
  answer to, run them through the app, and check how often the retrieved
  chunks actually contained the answer. That's a retrieval-accuracy metric
  you can put on your resume.
- **Show citations** — track which page each chunk came from, and display
  it next to the answer.
- **Support multiple PDFs** in one session, not just one at a time.
- **Try a different chunking strategy** — split on sentence or paragraph
  boundaries instead of a fixed character count, and see if answers improve.