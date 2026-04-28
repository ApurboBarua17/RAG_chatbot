import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_db"

app = FastAPI(title="CS Degree Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are a University of Arizona CS degree planning assistant.
Answer questions using ONLY the provided course catalog context.

Rules:
- Never make up course numbers, prerequisites, or requirements.
- If the answer is not in the context, say exactly:
  "I don't have that information — please check with your academic advisor at cs.arizona.edu/undergraduate/advising"
- Be concise and specific.
- Format course lists as bullet points.
- Always mention which document your answer came from.
"""


@app.on_event("startup")
async def startup_check():
    if not CHROMA_DIR.exists():
        print("ERROR: chroma_db not found. Run 'python ingest.py' first.", file=sys.stderr)
        sys.exit(1)


@app.get("/health")
def health():
    return {"status": "ok"}


# ── JSON endpoint (no file) ──────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request):
    content_type = request.headers.get("content-type", "")

    uploaded_file = None
    if "multipart/form-data" in content_type:
        form = await request.form()
        q = (form.get("question") or "").strip()
        uploaded_file = form.get("file")          # UploadFile or None
    else:
        body = await request.json()
        q = (body.get("question") or "").strip()

    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    from retriever import retrieve
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings

    # Retrieve from persistent Chroma
    chunks = retrieve(q)

    # If a PDF was uploaded, process it on-the-fly and prepend its chunks
    if uploaded_file and uploaded_file.filename:
        suffix = Path(file.filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            from langchain_community.vectorstores import Chroma as ChromaStore
            tmp_db = ChromaStore.from_documents(split_docs, embeddings)
            uploaded_chunks = tmp_db.similarity_search(q, k=3)
            file_chunks = [
                {"text": d.page_content, "source": uploaded_file.filename}
                for d in uploaded_chunks
            ]
            # Prepend uploaded doc results
            chunks = file_chunks + chunks[:2]
        finally:
            os.unlink(tmp_path)

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {q}",
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )

    answer = completion.choices[0].message.content
    return ChatResponse(answer=answer, sources=sources)
