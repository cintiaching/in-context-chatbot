from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.document_loaders import PyPDFLoader


def init_qa_chain(path, llm, chunk_size=500, chunk_overlap=20, prompt=None, doc_type="docx"):
    # load document
    if doc_type == "docx":
        loader = UnstructuredWordDocumentLoader(
            path, strategy="fast",
        )
    elif doc_type == "pdf":
        loader = PyPDFLoader(path)
    else:
        raise ValueError(f"{doc_type} format is not supported")

    docs = loader.load()

    # split the Document into chunks for embedding and vector storage.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)

    # store the splits to look up later
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
    retriever = vectorstore.as_retriever()

    # Prompt
    if prompt is None:
        prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. don't try to make up an answer"
            "Keep the answer concise, try to use exact wording from context that is relevant."
            "<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"
        )
    else:
        prompt = PromptTemplate.from_template(prompt)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain
