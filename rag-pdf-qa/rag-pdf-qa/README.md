# PDF Q&A (simple RAG)

Upload one PDF (for example 300 pages), ask questions, get answers with page citations.

- Backend: Python, FastAPI
- Embeddings: local `BAAI/bge-small-en-v1.5` via fastembed (no API key; downloads about 100 MB on first run)
- Search: cosine similarity with NumPy (no vector database needed)
- Answers: Groq API (key stays on the server)
- Frontend: one plain HTML file in `static/index.html`

## Setup
1. Install Python 3.10 or newer.
2. In this folder, create a virtual environment:
   `python -m venv .venv` then activate it
   (Windows: `.venv\Scripts\activate`, Mac/Linux: `source .venv/bin/activate`).
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and paste your Groq API key.
5. Check Groq's docs for a current model name and set `GROQ_MODEL` if needed.
6. Run: `uvicorn app:app --reload`
7. Open http://127.0.0.1:8000

## How it works
1. Upload: pypdf extracts text page by page.
2. Text is cut into chunks of about 1,000 characters (150 overlap). A chunk never crosses a page.
3. Each chunk becomes a vector. Vectors are saved in `data/` so a restart keeps your document.
4. Ask: your question becomes a vector, the 6 most similar chunks are found, and they are sent to the Groq model with the question.
5. The model answers only from those chunks and cites page numbers. The UI shows the source chunks.

## Limits
- One document at a time (uploading a new one replaces the old one).
- Scanned PDFs (images of text) need OCR first.
- Tables and multi-column layouts may extract in a messy order.
- Questions that need the whole document ("summarize everything") work poorly with top-6 retrieval.
- Retrieved text is sent to Groq, so do not upload confidential files without permission.

## Tuning
Edit the constants at the top of `app.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.
