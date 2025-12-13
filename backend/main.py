import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import asyncio

from dotenv import load_dotenv

load_dotenv()

from .ragpipe import index_documents, UPLOAD_DIR, query_qa, setConfig

app = FastAPI(title="RAG Backend")

config_store = {
    "azureEndpoint": None,
    "embeddingDeploymentName": None,
    "chatCompletionDeploymentName": None,
    "azureApiKey": None,
    "embedModelApiVersion": None,
    "chatCompletionModelApiVersion": None
}

origins = [
    "https://<your-angular-app-name>.azurestaticapps.net", # Your Angular URL
    "http://localhost:4200", # Optional: for local development testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        try:
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(dest)
        finally:
            f.file.close()

    # Index in background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, index_documents, saved_paths)

    return JSONResponse({"ok": True, "files": [os.path.basename(p) for p in saved_paths]})

@app.post("/query")
async def query(payload: dict):
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="'question' is required")

    answer, sources = await asyncio.get_event_loop().run_in_executor(None, query_qa, question)
    return {"answer": answer, "sources": sources}

@app.post("/config")
async def set_config(payload: dict):
    """Receive and store Azure configuration from the frontend."""
    try:
        # Validate that all required fields are present
        required_fields = [
            "azureEndpoint",
            "embeddingDeploymentName",
            "chatCompletionDeploymentName",
            "azureApiKey",
            "embedModelApiVersion",
            "chatCompletionModelApiVersion",
            "chromaDbCollectionName"
        ]
        
        for field in required_fields:
            if field not in payload or not payload[field]:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        config_store["azureEndpoint"] = payload["azureEndpoint"]
        config_store["embeddingDeploymentName"] = payload["embeddingDeploymentName"]
        config_store["chatCompletionDeploymentName"] = payload["chatCompletionDeploymentName"]
        config_store["azureApiKey"] = payload["azureApiKey"]
        config_store["embedModelApiVersion"] = payload["embedModelApiVersion"]
        config_store["chatCompletionModelApiVersion"] = payload["chatCompletionModelApiVersion"]
        config_store["chromaDbCollectionName"] = payload["chromaDbCollectionName"]

        os.environ["AZURE_OPENAI_ENDPOINT"] = config_store["azureEndpoint"]
        os.environ["OPENAI_API_KEY"] = config_store["azureApiKey"]
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = config_store["embedModelApiVersion"]
        os.environ["OPENAI_EMBED_DEPLOYMENT"] = config_store["embeddingDeploymentName"]
        os.environ["OPENAI_CHAT_DEPLOYMENT"] = config_store["chatCompletionDeploymentName"]
        os.environ["OPENAI_CHAT_API_VERSION"] = config_store["chatCompletionModelApiVersion"]
        os.environ["CHROMA_COLLECTION_NAME"] = config_store["chromaDbCollectionName"]
        
        return JSONResponse({
            "ok": True,
            "message": "Configuration saved successfully",
            "config": config_store
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Retrieve the current configuration."""
    return JSONResponse({"config": config_store})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
