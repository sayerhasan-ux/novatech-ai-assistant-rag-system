import sqlite3
from langchain_core.documents import Document

def load_customer_data(db_path: str):
    """
    Connects to the SQLite database, extracts records from all tables,
    and converts each row into a standarized LangChain document.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    documents = []

    tables = ["customers", "support_tickets"]
    for table_name in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]

        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        for index, row in enumerate(rows):
            row_text = ",".join(f"{col}: {val}" for col, val in zip(columns, row))

            doc = Document(
                page_content=row_text,
                metadata={
                    "source": db_path,
                    "table": table_name,
                    "row_id": index
                }
            )
            documents.append(doc)
    connection.close()
    return documents
if __name__ == "__main__":
    docs = load_customer_data("NovaTech_Customers.db")
    print(f"Successfully loaded {len(docs)} documents from Database.")
    print("--- Sample Document ---")
    print(docs[0].page_content)
    print(docs[0].metadata)

