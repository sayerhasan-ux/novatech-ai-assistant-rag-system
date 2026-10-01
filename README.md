# 🤖 NovaTech Solutions - AI Assistant (RAG System)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://novatech-ai-assistant-rag-system.streamlit.app)

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed for NovaTech Solutions to answer queries strictly based on internal company documents (Employee Handbooks, Sales Reports, and Customer Databases).

🔗 **Live Demo:** [NovaTech AI Assistant](https://novatech-ai-assistant-rag-system.streamlit.app)

---

## 🚀 Key Features
- **Multi-Source Ingestion:** Seamlessly chunks and indexes data across PDF policies, multi-sheet Excel workbooks, and SQLite database tables.
- **Local Semantic Search:** Powered by HuggingFace (`all-MiniLM-L6-v2`) embeddings and FAISS vector database (384-dimensional vector space).
- **Sub-Second Generation:** Integrated with Groq (`openai/gpt-oss-20b`) at low temperature (`0.2`) for deterministic, zero-hallucination responses.
- **Audit & Source Attribution:** Every answer provides expandable citations showing the source file and matched passage.
- **Conversational Memory:** Preserves multi-turn chat context across session interactions.
- **Interactive UI:** Built with Streamlit with one-click quick prompts.

---

## 🛠️ Tech Stack
- **Framework:** LangChain (LCEL)
- **LLM Engine:** Groq API (`openai/gpt-oss-20b`)
- **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Frontend & Deployment:** Streamlit Community Cloud

---

## 📦 Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sayerhasan-ux/novatech-ai-assistant-rag-system.git
   cd novatech-ai-assistant-rag-system
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Key:**

   Create a `.env` file in the root folder with:
   ```text
   GROQ_API_KEY=your_actual_groq_api_key
   ```
4. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

