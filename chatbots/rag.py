from abc import ABC
from chatbots.llama2_model import init_llama2_13b_llm
from chatbots.openai_model import init_openai_model
from langchain.document_loaders import UnstructuredWordDocumentLoader

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.document_loaders import PyPDFium2Loader


class DocumentChatbot(ABC):
    def __init__(self, doc_path: str, prompt_msg: str = None, model_name: str = "llama2_13b"):
        self.model_name = model_name
        self.doc_path = doc_path
        self.prompt_msg = prompt_msg
        self.llm = self.get_llm()
        self.docs = self.load_document()
        self.prompt = self.get_prompt()
        self.splits = self.get_splits()
        self.vectorstore = self.get_vectorstore()
        self.retriever = self.get_retriever()

    def get_llm(self):
        if self.model_name == "llama2_13b":
            llm = init_llama2_13b_llm()
        elif self.model_name == "openai":
            llm = init_openai_model()
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
        return llm

    def load_document(self):
        file_extension = self.doc_path.split(".")[-1]
        if file_extension == "docx":
            loader = UnstructuredWordDocumentLoader(
                self.doc_path, strategy="fast",
            )
        elif file_extension == "pdf":
            loader = PyPDFium2Loader(self.doc_path)
        else:
            raise ValueError(f"{file_extension} format is not supported")
        docs = loader.load()
        return docs

    def get_prompt(self):
        if self.prompt_msg is None:
            if self.model_name == "llama2_13b":
                prompt = PromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. don't try to make up an answer"
                    "try to use exact wording from context that is relevant, Keep the answer concise but give details"
                    "<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"
                )
            elif self.model_name == "openai":
                prompt = PromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "Don't try to make up an answer"
                    "Plz use exact wording from context that is relevant, and give details"
                    "\nQuestion: {question} \nContext: {context} "
                )
        else:
            prompt = PromptTemplate.from_template(self.prompt_msg)
        return prompt

    def qa_chain(self):
        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
        )
        return chain

    def get_splits(self):
        raise NotImplementedError

    def get_vectorstore(self):
        raise NotImplementedError

    def get_retriever(self):
        raise NotImplementedError
