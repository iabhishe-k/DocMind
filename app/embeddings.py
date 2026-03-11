from langchain_openai import OpenAIEmbeddings
from ingestion import load_and_chunk_pdf
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import re


load_dotenv()


def build_index(pdf_path: str):
        
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    chunks = load_and_chunk_pdf(pdf_path)

    QdrantVectorStore.from_documents(
        documents= chunks,
        embedding= embedding_model,
        url = "http://localhost:6333",
        collection_name = "learning_rag",
        force_recreate=True,
    )

    tokenized = [
        re.findall(r"\w+", doc.page_content.lower())
        for doc in chunks
    ]
    bm25 = BM25Okapi(tokenized)

    return chunks, bm25