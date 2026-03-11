from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()

TOP_K = 6
RRF_K = 60
embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

vector_db = QdrantVectorStore.from_existing_collection(
        embedding= embedding_model,
        url = "http://localhost:6333",
        collection_name = "learning_rag"
    )

retrieve_n = 20

def retriever(query: str, chunks: list, bm25: BM25Okapi, k: int = TOP_K):
    

    

    dense_results = vector_db.similarity_search_with_score(query= query, k= retrieve_n)

    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    

    top_bm25_indices = np.argsort(bm25_scores)[::-1][:retrieve_n]
    bm25_results = [
        (chunks[i], bm25_scores[i])
        for i in top_bm25_indices
    ]

    rrf_scores = {}

    for rank, (doc, _) in enumerate(dense_results, start=1):
        key = doc.page_content[:100] 
        rrf_scores[key] = rrf_scores.get(key, {"doc": doc, "score": 0.0})
        rrf_scores[key]["score"] += 1.0 / (RRF_K + rank)

    for rank, (doc, _) in enumerate(bm25_results, start=1):
        key = doc.page_content[:100]
        rrf_scores[key] = rrf_scores.get(key, {"doc": doc, "score": 0.0})
        rrf_scores[key]["score"] += 1.0 / (RRF_K + rank)

    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:k]

    return [item["doc"] for item in sorted_results]