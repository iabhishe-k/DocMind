from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


def retriever(user_query: str):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = QdrantVectorStore.from_existing_collection(
        embedding= embedding_model,
        url = "http://localhost:6333",
        collection_name = "learning_rag"
    )

    search_results = vector_db.similarity_search(query= user_query)

    return search_results