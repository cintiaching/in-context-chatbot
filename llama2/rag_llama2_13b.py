from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.document_loaders import PyPDFLoader

from utils import LLAMA2_13B_MODEL_PATH


def init_llama2_13b_llm(n_gpu_layers=1000, *kwarg):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # if the model is not stored locally, download from
    # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama
    llm = LlamaCpp(
        model_path=LLAMA2_13B_MODEL_PATH,
        n_gpu_layers=n_gpu_layers,  # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
        n_ctx=2048,  # Context size, text limits for responses
        # f16_kv MUST set to True, otherwise you will run into problem after a couple of calls
        # Use half-precision for key/value cache.
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=False,
        *kwarg
    )
    return llm


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