from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from app.rag_service import process_git_repo, process_uploaded_files, query_rag
import os
# Initialize the router
router = APIRouter()

# Define the query input structure
class QueryInput(BaseModel):
    query: str

# Route to upsert documents from a Git repository
@router.post("/upsert/git")
async def upsert_from_git(repo_url: str = Form(...), branch: str = Form("main")):
    return process_git_repo(repo_url, branch)

# Route to upsert documents from uploaded files
@router.post("/upsert/upload")
async def upsert_from_upload(file: UploadFile = File(...)):
    return process_uploaded_files(file)

# Route for querying embedded documents using RAG
@router.post("/query")
async def query_docs(query: QueryInput):
    return query_rag(query.query)
