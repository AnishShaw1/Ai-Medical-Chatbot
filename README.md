---

# 🩺 AI Medical Chatbot

An AI-powered medical assistant that provides evidence-based answers to health-related questions by retrieving information directly from verified medical documents. Built using LangChain, HuggingFace Transformers, FAISS, and Streamlit.

🔗 *Live Demo on Hugging Face Spaces*: [Try it here](https://huggingface.co/spaces/AnishShaw/ai-chat)

---

## 🚀 Features

- 💬 Asks and answers medical questions with contextual awareness
- 🔎 Retrieves answers from uploaded medical PDFs (no external hallucinations)
- 🧠 Uses Mistral 7B LLM via HuggingFace Inference API
- 🧾 Built-in RAG (Retrieval-Augmented Generation) with FAISS
- 🖥 Deployed on Hugging Face Spaces using Streamlit + Docker

---

## 🛠 Installation & Usage (Local)

> ✅ Recommended if you're testing or developing locally with Pipenv

1. *Clone the Repository*
   ```bash
   git clone https://github.com/AnishShaw1/Ai-Medical-Chatbot.git
   cd Ai-Medical-Chatbot

2. Set Your Environment Variable

Create a .env file in the root directory:

HF_TOKEN=your_huggingface_api_token


3. Install with Pipenv

pipenv install
pipenv shell


4. Run the App

streamlit run medibot.py




---

🐳 Deploying on Hugging Face Spaces

1. Make sure your repo includes:

medibot.py

requirements.txt

vectorstore/db_faiss with your FAISS index files



2. In Spaces, select:

SDK: Docker → Streamlit

Dockerfile should call medibot.py from /app



3. Add this to your Dockerfile if not already present:

ENTRYPOINT ["streamlit", "run", "medibot.py", "--server.port=8501", "--server.address=0.0.0.0"]


4. Ensure .env (with your HF token) is uploaded as a Secret in Hugging Face Space settings.




---

📚 Model & Stack

Component	Used Tool/Model

LLM	Mistral-7B-Instruct-v0.3 (via HuggingFace)
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Store	FAISS
Framework	LangChain
Interface	Streamlit



---

🛡 Disclaimer

> This chatbot is not a substitute for professional medical advice. Always consult with a licensed healthcare provider before making medical decisions.




---

📜 License

This project is licensed under the MIT License.


---

📌 Made by Anish Shaw

---
