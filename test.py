
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import TextLoader, PyPDFLoader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.chat_models import ChatOpenAI
from langchain_classic.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os, warnings, time

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def chunk_text(text, max_tokens=500, overlap=100):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_tokens:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + para

    if current:
        chunks.append(current.strip())

    return chunks

def create_chroma_db():
    documents = PyPDFLoader("The_Adventures_of_Tom_Sawyer.pdf").load()
    docs = split_docs(documents)
    print(type(documents), len(documents))
    print(type(docs), len(docs))    
    # 
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings, persist_directory="data")
    return db

def load_chroma_db():
    db = Chroma(
        persist_directory="data",
        embedding_function=OpenAIEmbeddings()
    )
    return db

if __name__ == "__main__":
    start_time = time.time() # 시작 시간 기록
    db = create_chroma_db()
    end_time = time.time() # 종료 시간 기록
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    llm = ChatOpenAI(model_name="gpt-4o")
    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        matching_docs = db.similarity_search(query)
        answer = chain.run(input_documents=matching_docs, question=query)
        print("AI: ", answer, "\n")
