import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    print("retrieving...")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone machine learning?"

    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
    )
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain = create_retrieval_chain(
    # retriever=vectorstore.as_retriever(),
    # combine_docs_chain=combine_docs_chain,
    # )
    # result = retrieval_chain.invoke(input={"input": query})
    # print(result)

    template = """answer the question with using the relevant context which is given to you. if you can not find the answer do  not try to make up an answer just say 'i could not find the answer from the context' .\n\ncontext: {context}\n\nquestion: {question}\n\nuse 5 sentences maximum and keep the answer as concise as possible. be kind and allways say "thanks for asking " at the end of the answer.
    {context}

    question: {question}

    answer:"""

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    # res = rag_chain.invoke(query)
    # print(res.content)
