from dotenv import load_dotenv
import os
from retriever import retriever
from openai import OpenAI
import time

load_dotenv()

def get_llm():
    return OpenAI()


class RAGPipeline:
    def __init__(self, retriever = retriever):
        self.retriever = retriever
        self.llm = get_llm()

    def answer(self, question: str) -> dict:
        
        results = self.retriever(question)

        context_parts = []
        sources = []
        chunks_text = []

        for doc in results:
            source_file = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page_label', '?')
            source_label = f"{source_file} (page {page_num})"

            context_parts.append(
                f"[Source: {source_label}]\n{doc.page_content}"
            )
            sources.append(source_label)
            chunks_text.append(doc.page_content)

        context = "\n\n---\n\n".join(context_parts)

        SYSTEM_PROMPT = f"""You are a precise document assistant. Answer questions using ONLY the information from the context below.

            Context:
            {context}

            Guidelines:
            - Answer using ONLY information from the context above
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

        