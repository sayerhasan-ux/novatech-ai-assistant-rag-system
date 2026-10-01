import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
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
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
    return vector_db, llm

vector_db, llm = load_rag_pipeline()


prompt_template = PromptTemplate(
    template="""You are a helpful AI assistant for NovaTech Solutions.
Answer the user's question based strictly on the following context. If you don't know the answer, say that you don't know.

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"]
)

rag_chain = prompt_template | llm | StrOutputParser()

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

user_query = st.chat_input("Ask a question about NovaTech...") or prompt_to_run

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Searching documents & thinking..."):
        matched_docs = vector_db.similarity_search(user_query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in matched_docs])
        response = rag_chain.invoke({"context": context_text, "question": user_query})

    with st.chat_message("assistant"):
        st.write(response)

    with st.expander("📄 View Reference Sources"):
        for i, doc in enumerate(matched_docs):
            source_name = doc.metadata.get("source", "Unknown")
            st.markdown(f"**Source {i+1}:** `{source_name}`")
            st.caption(doc.page_content[:250] + "...")

