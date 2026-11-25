import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
 
load_dotenv()  # Reads GROQ_API_KEY and MODEL_ID from the .env file
 
model = init_chat_model(
    model=os.getenv("MODEL_SMALL"),
    model_provider="groq",
)
 
company_doc = """
Nimbus Tech - Intern Handbook 2026
- Interns get 1 paid leave per month.
- Leave must be applied on the HR portal at least 3 days in advance.
- Unused leaves do not carry forward to the next month.
"""
 
 
def ask(question):
    # Without RAG: the AI answers only from its memory
    without_rag = model.invoke(question).content
 
    # With RAG: the AI answers using the handbook
    rag_prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply: "I don't know based on the given information."
 
Context:
{company_doc}
 
Question: {question}
"""
    with_rag = model.invoke(rag_prompt).content
    return without_rag, with_rag
 
 
print("Ask Nimbus Tech anything! Type 'exit' to quit.")
 
while True:
    question = input("\nYou: ").strip()
 
    if question.lower() in ("exit", "quit", "bye"):
        print("Goodbye!")
        break
    if not question:
        print("Please type a question.")
        continue
 
    without_rag, with_rag = ask(question)
    print("\nWITHOUT RAG:", without_rag)
    print("\nWITH RAG:", with_rag)
