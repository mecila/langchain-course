import os

from langchain_community.document_loaders import PyPDFLoader

os.environ.get("OPENAI_API_KEY")

if __name__ == "__main__":
    print("hi")
    pdf_path = "/home/acem/vectorstore-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
