# DocMind
### Conversational RAG Document Intelligence System

> Upload any PDF and have a full conversation with it — answers grounded strictly in the document, every response cited to the exact source page.

---

## Status: In Progress

| Component | Status |
|---|---|
| PDF Ingestion & Chunking | ✅ Done |
| Embedding Model (OpenAI text-embedding-3-large) | ✅ Done |
| Vector Store (Qdrant) | ✅ Done |
| RAG Pipeline | ✅ Done |
| RAGAS Evaluation | ✅ Done |
| Hybrid Search (Dense + BM25 + RRF) | 🔄 In Progress |
| Streamlit UI | 🔜 Coming |

---

## 📌 What is DocMind?

DocMind is a Retrieval-Augmented Generation (RAG) system that lets you upload any PDF document and ask questions about it in natural language. Instead of hallucinating answers, it retrieves the most relevant chunks from the document, passes them as context to an LLM, and returns a grounded answer with exact page citations.

**The problem it solves:** Large PDFs are hard to search through manually. DocMind lets you have a conversation with your document — ask follow-up questions, get precise answers, and always know which page the answer came from.

---

## 🛠 Tech Stack

| Layer | Tool | Why |
|---|---|---|
| PDF Loading | LangChain PyPDFLoader | Extracts text page by page with metadata |
| Chunking | RecursiveCharacterTextSplitter | Splits on paragraph → sentence → word boundaries |
| Embeddings | OpenAI text-embedding-3-large | 3072-dim, state-of-the-art semantic similarity |
| Vector Store | Qdrant (Docker) | Production-grade vector database |
| LLM | GPT-4o-mini | Fast, cheap, accurate for document QA |
| Orchestration | LangChain | Connects ingestion → retrieval → generation |
| Evaluation | RAGAS | Quantitative metrics for RAG quality |

---

## 🧠 How It Works

```
PDF File
    ↓
PyPDFLoader → extract text page by page
    ↓
RecursiveCharacterTextSplitter → 500 char chunks, 50 char overlap
    ↓
OpenAI text-embedding-3-large → 3072-dimensional vectors
    ↓
Qdrant → store vectors + metadata (source file, page number)
    ↓
User Query → embed query → cosine similarity search
    ↓
Top 4 chunks retrieved (LangChain default) → all passed as context to LLM
    ↓
GPT-4o-mini → grounded answer + page citation
```

---

## 🔍 Retrieval Algorithm

**Current: Dense Retrieval via Cosine Similarity**

Chunks are embedded using `text-embedding-3-large` (3072-dim). At query time, the question is embedded into the same space and Qdrant returns the top 4 most similar chunks via cosine similarity. All 4 chunks are passed as full context to GPT-4o-mini.

**Chunk size:** 500 characters, 50 character overlap

---


## 📊 RAGAS Evaluation

The pipeline is evaluated using [RAGAS](https://docs.ragas.io) on 20 hand-crafted QA pairs from the sample document.

| Metric | What it measures | Score (v1 — Dense only) |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | 0.971 |
| **Answer Relevancy** | Does the answer actually address the question? | 0.889 |
| **Context Recall** | Did retrieval find the chunks needed to answer correctly? | 0.717 |

Context recall at 0.717 is the weakest metric — expected with pure dense retrieval. Queries involving specific keywords, numbers, or proper nouns are where dense search underperforms. This is the primary motivation for adding hybrid search. Scores will be updated after each improvement.

Scores are computed once and saved to `data/eval/ragas_scores.json` — no repeated API calls on subsequent runs.

---

## 📁 File Structure

```
DocMind/
│
├── app/
│   ├── __init__.py
│   ├── embeddings.py        # PDF ingestion + OpenAI embedding + Qdrant indexing
│   ├── ingestion.py         # PDF loading and chunking
│   ├── retriever.py         # Qdrant similarity search
│   ├── rag_pipeline.py      # Core RAG chain: retrieve → prompt → LLM → answer
│   └── evaluator.py         # RAGAS evaluation suite
│
├── data/
│   ├── dsa.pdf              # Sample document (never committed)
│   └── eval/
│       ├── qa_pairs.json        # 20 QA pairs for evaluation(auto-generated)
│
├── .env                     # API keys (never committed)
├── .env.example             # Safe template
├── .gitignore
├── docker-compose.yml       # Qdrant vector database
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Docker Desktop

### 1. Clone the repository
```bash
git clone https://github.com/iabhishe-k/docmind.git
cd docmind
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
copy .env.example .env
```
Edit `.env` and add your keys:
```
OPENAI_API_KEY=your_openai_api_key_here
```
Get your OpenAI API key at [platform.openai.com](https://platform.openai.com)

### 5. Start Qdrant
```bash
docker-compose up -d
```
Verify at [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### 6. Index your document
```bash
cd app
python embeddings.py
```
This loads `data/dsa.pdf`, chunks it, embeds it, and stores vectors in Qdrant.

### 7. Run the pipeline
```bash
python rag_pipeline.py
```

### 8. Run evaluation (optional)
```bash
python evaluator.py
```
Scores are saved after the first run — subsequent runs load from cache.

---

## 🐳 Docker Commands

```bash
docker-compose up -d      # start Qdrant
docker-compose down       # stop Qdrant
docker-compose logs       # view logs
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | For embeddings (text-embedding-3-large) and LLM (gpt-4o-mini) |

---

## 👤 Author

**Abhishek Kumar Singh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/iabhishe-k)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-blue)](https://github.com/iabhishe-k)