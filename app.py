import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


st.set_page_config(page_title="NovaTech AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 NovaTech Solutions - AI Assistant")
st.caption("Ask questions about company policies, sales performance, or employee handbook.")

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key and "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
        
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_key,
        temperature=0.2,
        max_tokens=500
    )
    return vector_db, llm

vector_db, llm = load_rag_pipeline()

prompt_template = PromptTemplate(
    template="""You are a helpful AI assistant for NovaTech Solutions.
Answer the user's question based strictly on the following document context and recent conversation history. 
If the answer cannot be determined from the context, state that clearly without guessing.

Conversation History:
{chat_history}

Document Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["chat_history", "context", "question"]
)

rag_chain = prompt_template | llm | StrOutputParser()


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown("##### 💡 Try asking:")
col1, col2, col3 = st.columns(3)

prompt_to_run = None
with col1:
    if st.button("📋 Leave Policy", use_container_width=True):
        prompt_to_run = "What is the employee leave policy and annual allowance?"
with col2:
    if st.button("📊 Sales Top Performer", use_container_width=True):
        prompt_to_run = "Who are the top sales performers and in which regions?"
with col3:
    if st.button("🎫 Support Tickets", use_container_width=True):
        prompt_to_run = "What are the common support ticket issues logged by customers?"


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 View Reference Sources"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source {i+1}:** `{src['source']}`")
                    st.caption(src["content"][:250] + "...")


user_query = st.chat_input("Ask a question about NovaTech...") or prompt_to_run

if user_query:
    
    with st.chat_message("user"):
        st.write(user_query)

    
    history_lines = [
        f"{m['role'].capitalize()}: {m['content']}" 
        for m in st.session_state.messages[-4:]
    ]
    chat_history_str = "\n".join(history_lines) if history_lines else "None"

    # Search documents and run the RAG chain
    with st.spinner("Searching documents & thinking..."):
        matched_docs = vector_db.similarity_search(user_query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in matched_docs])
        response = rag_chain.invoke({
            "chat_history": chat_history_str,
            "context": context_text,
            "question": user_query
        })

    with st.chat_message("assistant"):
        st.write(response)

    sources_data = [
        {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
        for doc in matched_docs
    ]

    with st.expander("📄 View Reference Sources"):
        for i, src in enumerate(sources_data):
            st.markdown(f"**Source {i+1}:** `{src['source']}`")
            st.caption(src["content"][:250] + "...")

   
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources_data})