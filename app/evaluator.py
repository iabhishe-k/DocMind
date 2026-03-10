from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from datasets import Dataset
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import json
import time
from pathlib import Path


load_dotenv()

QA_PAIRS_PATH = Path(__file__).parent.parent / "data/eval/qa_pairs.json"

run_config = RunConfig(
    max_workers=4,
    timeout=180
)


def load_qa_pairs() -> list:
    with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

pipeline = RAGPipeline()
qa_pairs = load_qa_pairs()

questions, answers, contexts, ground_truths = [], [], [], []

for i, pair in enumerate(qa_pairs):
    print(f"Question {i+1}/{len(qa_pairs)}: {pair['question'][:50]}...")
    result = pipeline.answer(pair["question"])

    questions.append(pair["question"])
    answers.append(result["answer"])
    contexts.append(result["chunks"])
    ground_truths.append(pair["ground_truth"])

eval_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truths,
})

async_client = AsyncOpenAI()

ragas_llm = llm_factory(
    model="gpt-4o-mini",
    client=async_client,
    max_tokens = 4096
)
ragas_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall
]

result = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config
)

scores = {
    "faithfulness": float(round(sum(result["faithfulness"]) / len(result["faithfulness"]), 3)),
    "answer_relevancy": float(round(sum(result["answer_relevancy"]) / len(result["answer_relevancy"]), 3)),
    "context_recall": float(round(sum(result["context_recall"]) / len(result["context_recall"]), 3)),
}

print(f"[evaluator] Results: {scores}")