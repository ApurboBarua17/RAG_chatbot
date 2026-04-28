import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


def ingest():
    pdf_files  = list(DATA_DIR.glob("*.pdf"))
    txt_files  = list(DATA_DIR.glob("*.txt"))
    all_files  = pdf_files + txt_files

    if not all_files:
        print(f"No PDFs or .txt files found in {DATA_DIR}. Add documents and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) and {len(txt_files)} text file(s):")
    for f in all_files:
        print(f"  - {f.name}")

    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    for txt in txt_files:
        loader = TextLoader(str(txt), encoding="utf-8")
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} document sections total.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Saved vector store to {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
