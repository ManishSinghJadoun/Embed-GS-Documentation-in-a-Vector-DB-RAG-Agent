import os
import tempfile
import git
from transformers import AutoTokenizer, AutoModel
import torch
import faiss

# Initialize the TinyLlama model (a smaller variant of Llama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# In-memory storage for documents and their embeddings
DOC_STORE = []
EMBEDDINGS = []
index = None

# Function to embed text using TinyLlama model
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :].squeeze().numpy()

# Function to upsert text into the vector database (Faiss)
def upsert_text(text):
    global index
    emb = embed_text(text)
    EMBEDDINGS.append(emb)
    DOC_STORE.append(text)
    if index is None:
        index = faiss.IndexFlatL2(len(emb))  # Initialize Faiss index
    index.add(emb.reshape(1, -1))  # Add embedding to Faiss index

# Function to extract markdown files from a Git repository
def extract_markdown_files(repo_dir):
    texts = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
    return texts

# Function to process Git repository and upsert documents
def process_git_repo(repo_url, branch):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Clone the repository
        git.Repo.clone_from(repo_url, tmpdirname, branch=branch)
        docs = extract_markdown_files(tmpdirname)  # Extract docs
        for doc in docs:
            upsert_text(doc)  # Upsert each document
    return {"status": "success", "docs_upserted": len(docs)}

# Function to process uploaded files and upsert text
def process_uploaded_files(file):
    text = file.file.read().decode("utf-8")
    upsert_text(text)
    return {"status": "success", "docs_upserted": 1}

# Function to query the embedded documents using RAG
def query_rag(query):
    if index is None or not DOC_STORE:
        return {"error": "No documents in database."}
    q_emb = embed_text(query)
    D, I = index.search(q_emb.reshape(1, -1), 3)  # Search top 3 matches
    results = [DOC_STORE[i] for i in I[0] if i < len(DOC_STORE)]
    return {"query": query, "matches": results}
