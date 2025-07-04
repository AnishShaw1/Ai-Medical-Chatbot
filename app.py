from flask import Flask, render_template, request, session
from functools import lru_cache
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os

app = Flask(__name__)

# Ensure session expires when browser closes
app.config["SESSION_PERMANENT"] = False

app.secret_key = os.environ.get("FLASK_SECRET_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"

@lru_cache(maxsize=1)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)


@app.route("/", methods=["GET", "POST"])
def index():
    #session.clear()

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form["query"]

        CUSTOM_PROMPT_TEMPLATE = """
        You are a helpful, cautious, and evidence-based medical assistant.

        Use ONLY the information provided in the context below to answer the user's question. Do not use any external knowledge. If the answer is not clearly stated in the context, respond with "I'm not sure based on the provided information."

        Keep your answer:
        - Concise
        - Fact-based
        - Free of speculation
        - Without unnecessary small talk

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': query})
            result = response["result"]
            source_documents = response["source_documents"]

            sources_formatted = [
                {
                    "page": doc.metadata.get("page", "N/A"),
                    "file": doc.metadata.get("source", ""),
                    "snippet": doc.page_content[:500]
                }
                for doc in source_documents
            ]

            session["chat_history"].append({
                "user": query,
                "bot": result,
                "sources": sources_formatted
            })

            session.modified = True  # Inform Flask session was updated

        except Exception as e:
            session["chat_history"].append({
                "user": query,
                "bot": f"❌ Error: {str(e)}",
                "sources": []
            })
            

    return render_template("index.html", chat_history=session.get("chat_history", []))


if __name__ == "__main__":
    app.run(debug=True)
