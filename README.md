# German Grammar RAG Tutor

A German grammar tutor that retrieves relevant passages from a PDF knowledge base and generates explanations with OpenAI. It includes a Streamlit chat interface, Qdrant vector storage, duplicate-chunk filtering, and source references showing both printed book labels and PDF viewer page numbers.

## Requirements

- Python 3.9 or newer
- Docker with Docker Compose
- An OpenAI API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and add your OpenAI API key.
4. Start Qdrant with `docker compose up -d`.
5. Index the included PDF with `python index.py`.
6. Launch the tutor with `streamlit run app.py` and open the local URL shown by Streamlit.

To delete and recreate the vector collection, run `python index.py --rebuild`. This re-embeds the PDF and may incur OpenAI API usage costs.

## Project files

- `index.py` loads and chunks the PDF, creates embeddings, and stores them in Qdrant.
- `dedupe.py` removes duplicate chunks and retrieval results.
- `retrive.py` provides a command-line retrieval and answer flow.
- `app.py` provides the Streamlit chat interface and source references.
- `docker-compose.yml` runs the Qdrant vector database.

The OpenAI and optional Qdrant API keys are read from `.env`; do not commit real credentials.

To index the PDF into Qdrant Cloud, set `OPENAI_API_KEY`, `QDRANT_URL`, and
`QDRANT_API_KEY` in your ignored local `.env` file, then run `python index.py`.
The indexer uploads the PDF's extracted text chunks and embeddings to Qdrant;
the original PDF remains a separate file and is not stored as a PDF inside the
vector database. After indexing, configure the same Qdrant URL and API key in
Streamlit Cloud app secrets so the deployed tutor can retrieve those chunks.

## Streamlit Community Cloud

In the deployed app, open **Manage app → Settings → Secrets** and add the
following TOML values using your own credentials:

```toml
OPENAI_API_KEY = "your-openai-api-key"
QDRANT_URL = "https://your-qdrant-instance"
QDRANT_API_KEY = "your-qdrant-api-key"
```

`QDRANT_API_KEY` may be omitted if your hosted Qdrant instance does not require
one. A cloud-hosted app cannot connect to `localhost` on your computer, so
`QDRANT_URL` must point to a reachable hosted Qdrant instance. The collection
must also be populated with the indexed PDF chunks.
