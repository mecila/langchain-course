import os

os.environ.get("OPENAI_API_KEY")
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

if __name__ == "__main__":
    print("hi")
    pdf_path = "/home/acem/vectorstore-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    texts = splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index")
    new_vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    document_chain = create_stuff_documents_chain(
        llm=OpenAI(temperature=0), prompt=retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), document_chain
    )

    result = retrieval_chain.invoke(
        {"input": "give me the gist of ReAct in 5 sentences"}
    )
    print(result["answer"])
