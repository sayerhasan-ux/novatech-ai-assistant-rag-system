from langchain_community.document_loaders import PyPDFLoader
def load_handbook(pdf_path: str):
    """
    Loads a PDF and returns a list of LangChain Document objects - 
    one Document per page.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    docs = load_handbook("NovaTech_Employee_Handbook.pdf")

    print(f"Load {len(docs)} page(s)\n")

    first = docs[0]
    print("--- page_content (firat 300 chars) ---")
    print(first.page_content[:300])
    print("\n--- metadata ---")
    print(first.metadata)
