import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

from langchain_huggingface.chat_models import ChatHuggingFace

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",  # REQUIRED for Mistral
        temperature=0.5,
        max_new_tokens=512
    )

    chat_llm = ChatHuggingFace(llm=llm)  # ✅ Wrap the LLM
    return chat_llm



# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful, cautious, and evidence-based medical assistant.

Use ONLY the information provided in the context below to answer the user's question. Do not use any external knowledge. If the answer is not clearly stated in the context, respond with "I'm not sure based on the provided information."

Keep your answer:
- Concise
- Fact-based
- Free of speculation
- Without unnecessary small talk

Present your answer in a well-structured format if the context supports it (e.g., bullets, steps, summary).

Context:
{context}

Question:
{question}

Answer:

"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:")

for i, doc in enumerate(response["source_documents"], start=1):
    print(f"\n--- Source #{i} ---")
    print(f"📄 Page: {doc.metadata.get('page_label', 'Unknown')}")
    print(f"📁 File: {doc.metadata.get('source', 'Unknown')}")
    print("🧾 Content Snippet:\n", doc.page_content.strip())
