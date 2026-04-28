from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = Path(__file__).parent / "chroma_db"


def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    return db


def retrieve(query: str, k: int = 4) -> list[dict]:
    db = get_retriever()
    results = db.similarity_search(query, k=k)
    return [
        {
            "text": doc.page_content,
            "source": Path(doc.metadata.get("source", "unknown")).name,
        }
        for doc in results
    ]
