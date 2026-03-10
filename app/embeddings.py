from langchain_openai import OpenAIEmbeddings
from ingestion import load_and_chunk_pdf
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent.parent / "data/dsa.pdf"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

chunks = load_and_chunk_pdf(pdf_path)

vector_store = QdrantVectorStore.from_documents(
    documents= chunks,
    embedding= embedding_model,
    url = "http://localhost:6333",
    collection_name = "learning_rag"
)