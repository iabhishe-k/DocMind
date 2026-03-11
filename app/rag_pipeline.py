from dotenv import load_dotenv
from retriever import retriever
from embeddings import build_index
from openai import OpenAI
from pathlib import Path

load_dotenv()
PDF_PATH = Path(__file__).parent.parent / "data/dsa.pdf"


class RAGPipeline:
    def __init__(self):
        self.chunks, self.bm25 = build_index(PDF_PATH)
        self.llm = OpenAI()

    def answer(self, question: str) -> dict:
        
        results = retriever(question, self.chunks, self.bm25)

        context_parts = []
        sources = []
        chunks_text = []

        for doc in results:
            source_file = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page_label', '?')
            source_label = f"{source_file} (page {page_num})"
            context_parts.append(f"[Source: {source_label}]\n{doc.page_content}")
            sources.append(source_label)
            chunks_text.append(doc.page_content)

        context = "\n\n---\n\n".join(context_parts)

        SYSTEM_PROMPT = f"""You are a precise document assistant. Answer questions using ONLY the information from the context below.

            Context:
            {context}

            Guidelines:
            - Carefully read all context chunks before answering.
            - If multiple chunks contain relevant information, combine them.
            - Always mention the page number your answer comes from
            - If the answer is not in the context, respond exactly:
            "I could not find this information in the provided document."
            - Be concise and direct

            Format your response as:

            Answer:
            <your explanation here>

            Source:
            Page <page_number>
        """

        
        # Step 5: Call the LLM
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            max_completion_tokens=200,
            temperature= 0.2
        )

        answer_text = response.choices[0].message.content

        return {
            "answer": answer_text,
            "sources": list(set(sources)),
            "chunks": chunks_text,
            "question": question,
        }

        