# DocMind

### Conversational RAG Document Intelligence System

> Upload any PDF and have a full conversation with it — answers grounded strictly in the document with explicit page citations.

---

## 🚧 Status

| Component                | Status     |
| ------------------------ | ---------- |
| PDF Ingestion & Chunking | ✅ Done    |
| Embedding Model          | ✅ Done    |
| Vector Store (Qdrant)    | ✅ Done    |
| RAG Pipeline             | ✅ Done    |
| RAGAS Evaluation         | ✅ Done    |
| Hybrid Retrieval         | ✅ Done    |
| Streamlit UI             | 🔜 Coming  |

---

## 📌 What is DocMind?

DocMind is a **Retrieval-Augmented Generation (RAG) system** that allows users to upload a PDF and ask questions about it using natural language.

Instead of relying purely on an LLM, DocMind retrieves the most relevant sections from the document and provides them as context to the model. The final answer is generated **strictly from retrieved content**, with **source page citations**.

### Problem

Large PDFs (technical papers, books, reports) are difficult to navigate manually.

DocMind allows users to:

- Ask questions about a document
- Get grounded answers
- See exactly **which page the answer came from**

---

## 🛠 Tech Stack

| Layer           | Tool                            | Purpose                       |
| --------------- | ------------------------------- | ----------------------------- |
| PDF Loading     | LangChain `PyPDFLoader`         | Extract page-wise text        |
| Chunking        | RecursiveCharacterTextSplitter  | Semantic chunk splitting      |
| Embeddings      | OpenAI `text-embedding-3-large` | High quality semantic vectors |
| Vector Database | Qdrant (Docker)                 | Vector similarity search      |
| LLM             | `gpt-4o-mini`                   | Answer generation             |
| Retrieval       | Dense + BM25 + RRF              | Hybrid search                 |
| Evaluation      | RAGAS                           | Quantitative RAG evaluation   |
| UI (Planned)    | Streamlit                       | Interactive interface         |

---

## 🧠 System Architecture

```
PDF Document
     ↓
PyPDFLoader
     ↓
RecursiveCharacterTextSplitter
(800 char chunks, 150 overlap)
     ↓
OpenAI Embeddings
(text-embedding-3-large)
     ↓
Qdrant Vector Store
     ↓
User Query
     ↓
Dense Retrieval (20 candidates)
+
BM25 Retrieval (20 candidates)
     ↓
Reciprocal Rank Fusion
     ↓
Top 6 Context Chunks
     ↓
GPT-4o-mini
     ↓
Grounded Answer + Page Citations
```

---

## 🔍 Retrieval Strategy Evolution

DocMind went through multiple iterations to improve **retrieval quality and answer grounding**.

---

### v1 — Dense Retrieval

Initial version used pure vector similarity search.

```
Query → Embedding → Qdrant similarity → Top 4 → LLM
```

| Parameter        | Value          |
| ---------------- | -------------- |
| Chunk size       | 500 characters |
| Overlap          | 50 characters  |
| Retrieved chunks | 4              |

#### Metrics

| Metric           | Score |
| ---------------- | ----- |
| Faithfulness     | 0.971 |
| Answer Relevancy | 0.889 |
| Context Recall   | 0.717 |

---

### v2 — Hybrid Retrieval

Hybrid retrieval combines **semantic similarity** with **keyword matching**.

Dense retrieval captures semantic meaning while BM25 helps retrieve:

- proper nouns
- numbers
- technical keywords

Results from both retrievers are merged using **Reciprocal Rank Fusion (RRF)**.

```
Dense (8)
+
BM25 (8)
↓
RRF
↓
Top 4
```

#### RRF Formula

```
score(d) = Σ 1 / (k + rank_i)
```

where `k = 60`.

#### Metrics

| Metric           | Score |
| ---------------- | ----- |
| Faithfulness     | 0.928 |
| Answer Relevancy | 0.894 |
| Context Recall   | 0.692 |

Hybrid retrieval improved answer relevancy but slightly reduced context recall due to noisy BM25 matches on technical text.

---

### v3 — Improved Hybrid Retrieval (Final)

Two improvements were introduced:

1. Larger chunk size for better semantic completeness
2. Larger candidate pool before RRF fusion

```
Dense (20)
+
BM25 (20)
↓
RRF
↓
Top 6
↓
LLM
```

#### Parameters

| Parameter  | Value          |
| ---------- | -------------- |
| Chunk size | 800 characters |
| Overlap    | 150 characters |
| retrieve_n | 20             |
| top_k      | 6              |

#### Metrics

| Metric           | Score     |
| ---------------- | --------- |
| Faithfulness     | 0.946     |
| Answer Relevancy | 0.905     |
| Context Recall   | **0.846** |

---

## 📊 RAGAS Evaluation

The system is evaluated using **RAGAS** on a dataset of **50 manually created QA pairs** derived from the sample document.

Metrics used:

| Metric           | Meaning                                              |
| ---------------- | ---------------------------------------------------- |
| Faithfulness     | Whether the answer is supported by retrieved context |
| Answer Relevancy | Whether the answer addresses the question            |
| Context Recall   | Whether the retriever fetched the correct context    |

### Final Results

| Metric           | Dense | Hybrid | Improved Hybrid |
| ---------------- | ----- | ------ | --------------- |
| Faithfulness     | 0.971 | 0.928  | 0.946           |
| Answer Relevancy | 0.889 | 0.894  | 0.905           |
| Context Recall   | 0.717 | 0.692  | **0.846**       |

---

## 📈 Key Insights

- **Chunk size strongly affects recall** — larger chunks preserve more context.
- **Hybrid retrieval needs larger candidate pools** for RRF to work effectively.
- **Dense retrieval alone performs very well**, but hybrid search improves coverage.
- **RAGAS evaluation prevented regression** — intuition alone suggested v2 was better, but metrics showed otherwise.

---

## 📁 Project Structure

```
DocMind/
│
├── app/
│   ├── __init__.py
│   ├── embeddings.py        
│   ├── ingestion.py         
│   ├── retriever.py         
│   ├── rag_pipeline.py    
│   └── evaluator.py        
│
├── data/
│   ├── dsa.pdf                 (never committed)
│   └── eval/
│       └── qa_pairs.json        
│
├── .env                        (never committed)
├── .env.example             
├── .gitignore
├── docker-compose.yml     
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone https://github.com/iabhishe-k/docmind.git
cd docmind
```

### 2. Create virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

```bash
copy .env.example .env
```

Add your OpenAI key:

```
OPENAI_API_KEY=your_key_here
```

### 5. Start Qdrant

```bash
docker-compose up -d
```

Dashboard: `http://localhost:6333/dashboard`

### 6. Index a document

Place a PDF inside `data/` and update the path inside the pipeline if needed.

### 7. Run the RAG pipeline

```bash
python app/rag_pipeline.py
```

### 8. Run evaluation

```bash
python app/evaluator.py
```

---

## 🐳 Docker Commands

```bash
# Start Qdrant
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs
```

---

## 🔑 Environment Variables

| Variable         | Description                            |
| ---------------- | -------------------------------------- |
| `OPENAI_API_KEY` | Used for embeddings and LLM generation |

---

## 👤 Author

**Abhishek Kumar Singh**  


[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/iabhishe-k)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-blue)](https://github.com/iabhishe-k)