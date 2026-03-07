from dotenv import load_dotenv
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from retriever import retriever
from openai import OpenAI

load_dotenv()




client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

user_query = input("Ask me anything: ")

search_results = retriever(user_query=user_query)

context = "\n\n\n".join([
    f"Page Content : {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" 
    for result in search_results
])


SYSTEM_PROMPT = f"""
You are an AI assistant helping users understand information from a PDF document.

Use ONLY the provided context to answer the question.

Guidelines:
- If the answer exists in the context, explain it clearly.
- Include the page number where the information was found.
- If the answer is not in the context, respond:
  "I could not find this information in the provided document."

Context:
{context}

When answering, follow this format:

Answer:
<explanation>

Source:
Page <page_number>
"""


response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {   "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ],
    temperature=0.2
)

print(response.choices[0].message.content)