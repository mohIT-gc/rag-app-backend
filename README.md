# RAG Backend (FastAPI)

This backend provides two endpoints:

- `POST /upload` — upload one or more files to be indexed
- `POST /query` — ask a question and get an answer with sources

Setup

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill your credentials (Azure OpenAI endpoint, key, etc.)

3. Run the server:

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Notes

- This starter uses local FAISS vectorstore for development. For Azure deployment replace the vector store with Azure Cognitive Search.

