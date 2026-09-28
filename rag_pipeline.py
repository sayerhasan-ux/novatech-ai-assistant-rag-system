from main import load_handbook
from load_excel import load_sales_data
from load_db import load_customer_data
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

pdf_docs = load_handbook("NovaTech_Employee_Handbook.pdf")
excel_docs = load_sales_data("NovaTech_Sales_Data.xlsx")
db_docs = load_customer_data("NovaTech_Customers.db")

all_documents = pdf_docs + excel_docs + db_docs
print(f"All documents loaded across all sources: {len(all_documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60
)
chunks = text_splitter.split_documents(all_documents)

print(f"Total chunks created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
sample_vector = embeddings.embed_query(chunks[0].page_content)
print(f"Embedding successful! Vector dimentions: {len(sample_vector)}")

print("Indexing chunks into FAISS vector database...")
vector_db = FAISS.from_documents(chunks, embeddings)

vector_db.save_local("faiss_index")
print("Vector database successfully built and saved in 'faiss_index'!")

query = "employee leave policy"  # ya sales se related koi sawal

# Top 2 most relevant chunks dhoondein
matched_docs = vector_db.similarity_search(query, k=2)

print("\n--- Search Results ---")
for i, doc in enumerate(matched_docs):
    print(f"\nResult {i+1} [Source: {doc.metadata.get('source')}]:")
    print(doc.page_content[:200] + "...")

context_text = "\n\n".join([doc.page_content for doc in matched_docs])
prompt_template = PromptTemplate(
    template = """You are a helpful AI assistant for NovaTech Solutions. 
    Answer the user's questions based strictly on the following context. If you don't know the answer, say that you don't know.
Context:
{context}

Question:
{question}

Answer:
    """,
    input_variables=["context", "question"]
)

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
rag_chain = prompt_template | llm | StrOutputParser()

print("\Generating final answer from Gemini LLM...")
final_answer = rag_chain.invoke({"context": context_text, "question": query})

print("\n--- Final AI Answer ---")
print(final_answer)




