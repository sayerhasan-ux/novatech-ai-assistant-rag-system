import pandas as pd
from langchain_core.documents import Document
def load_sales_data(excel_path: str):
    """
    Loads all sheets from an Excel workbook and converts each row into a standarized
    LangChain Document object for vector embedding.
    """
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    documents = []

    for sheet_name, df in all_sheets.items():
        for index, row in df.iterrows():
            row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
            doc = Document(
                page_content=row_text,
                metadata={
                    "source": excel_path,
                    "sheet": sheet_name,
                    "row": index
                }
            )
            documents.append(doc)
    return documents

if __name__ == "__main__":
    docs = load_sales_data("NovaTech_Sales_Data.xlsx")
    print(f"Loaded {len(docs)} documents successfully!")



