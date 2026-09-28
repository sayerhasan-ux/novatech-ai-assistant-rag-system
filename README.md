# NovaTech Solutions - AI Assistant (RAG System)

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed for NovaTech Solutions to answer queries strictly based on internal company documents (Employee Handbooks, Sales Reports, and Customer Databases).

---

## Key Features
- **Multi-Source Ingestion:** Seamlessly chunks and indexes data across PDF policies, Excel sheets, and databases.
- **Local Semantic Search:** Powered by HuggingFace (`all-MiniLM-L6-v2`) embeddings and FAISS vector database (384-dimensional vector space).
- **Strict Factual Generation:** Integrated with Google Gemini (`gemini-3.8-flash`) using a low temperature (`0.2`) to prevent hallucinations.
- **Reference Tracking:** Every answer includes exact source file citations for auditing and transparency.
- **Interactive UI:** Built with Streamlit for a fast, responsive chat experience.

---

## Tech Stack
- **Framework:** LangChain (LCEL Pipeline)
- **LLM:** Google Gemini (`gemini-3.8-flash`)
- **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Frontend:** Streamlit

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sayerhasan-ux/novatech-ai-assistant.git
   cd novatech-ai-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key:**
   Create a `.env` file in the root folder:
   ```text
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
