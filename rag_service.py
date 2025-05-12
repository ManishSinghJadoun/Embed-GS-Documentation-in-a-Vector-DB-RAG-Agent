import os
import tempfile
import git
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss

# Initialize the TinyLlama model (a smaller variant of Llama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# In-memory storage for documents and their embeddings
DOC_STORE = []
EMBEDDINGS = []
index = None
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    return embed_model.encode(text)

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

    # Step 1: Embed the query
    q_emb = embed_text(query)
    D, I = index.search(q_emb.reshape(1, -1), 1)  # Get top 1 relevant document

    if not I[0].size or I[0][0] >= len(DOC_STORE):
        return {"error": "No relevant documents found."}

    # Step 2: Retrieve top document
    context = DOC_STORE[I[0][0]]

    # Step 3: Construct prompt for TinyLlama
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Step 4: Tokenize and generate response
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Step 5: Extract generated answer from output
    answer = response.split("Answer:")[-1].strip()

    return {
        "query": query,
        "matched_doc": context,
        "llm_answer": answer
    }
