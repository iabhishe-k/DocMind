from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ingestion import load_and_chunk_pdf
from langchain_qdrant import QdrantVectorStore
from pathlib import Path

pdf_path = Path(__file__).parent.parent / "data/dsa.pdf"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chunks = load_and_chunk_pdf(pdf_path)

vector_store = QdrantVectorStore.from_documents(
    documents= chunks,
    embedding= embedding_model,
    url = "http://localhost:6333",
    collection_name = "learning_rag"
)