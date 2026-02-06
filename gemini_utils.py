import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a PDF assistant.
Answer ONLY from the context.
If answer is not present, say "Not found in PDF".

Context:
{context}

Question:
{question}
"""
    response = model.generate_content(prompt)
    return response.text








