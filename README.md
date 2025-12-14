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

2. Run the server:

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Notes

- If Angular UI deployed into new Azure Static Web App, configure the frontend url in main.py origins section.
  by default, keep the origins unchanged [https://purple-meadow-08d99360f.3.azurestaticapps.net]
```powershell
origins = [
    "https://{your_angular_ui_app_name}.azurestaticapps.net", # Your Angular URL
    ......
]
```

The project code  can be accessed through github [https://github.com/mohIT-gc/rag-app-backend]
Deployed Azure Web App for backend service URL
[https://gmuece553-team4-rag-backend-a2fghsejh3awbncp.canadacentral-01.azurewebsites.net]
