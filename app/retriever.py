from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

def retriever(user_query: str):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vector_db = QdrantVectorStore.from_existing_collection(
        embedding= embedding_model,
        url = "http://localhost:6333",
        collection_name = "learning_rag"
    )

    search_results = vector_db.similarity_search(query= user_query)

    return search_results