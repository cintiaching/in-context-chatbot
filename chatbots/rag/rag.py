import os
from abc import ABC, abstractmethod
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFium2Loader

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from chatbots.llm.llm import LLMs, LLMConfig, LLMFactory


class DocumentChatbot(ABC):
    def __init__(self, model_name: LLMs, doc_path: str, collection_name: str, persist_directory: str,
                 prompt_msg: str = None):
        self.model_name = model_name
        self.doc_path = doc_path
        self.prompt_msg = prompt_msg
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.llm = self.get_llm()
        self.docs = self.load_document()
        self.prompt = self.get_prompt()
        self.splits = self.get_splits()
        self.vectorstore = self.get_vectorstore()
        self.retriever = self.get_retriever()

    def get_llm(self):
        config = LLMConfig(model_name=self.model_name)
        return LLMFactory.initiate_llm(config)

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
        if self.prompt_msg:
            prompt = PromptTemplate.from_template(self.prompt_msg)
        else:
            # use default prompt
            if self.model_name == LLMs.LLAMA2_13B:
                prompt = PromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. don't try to make up an answer"
                    "try to use exact wording from context that is relevant, Keep the answer concise but give details"
                    "<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"
                )
            elif self.model_name == LLMs.GPT_3_PT_5_TURBO:
                prompt = PromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "Don't try to make up an answer"
                    "Plz use exact wording from context that is relevant, and give details"
                    "\nQuestion: {question} \nContext: {context} "
                )
            else:
                prompt = PromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. don't try to make up an answer"
                    "try to use exact wording from context that is relevant, Keep the answer concise but give details"
                    "\nQuestion: {question} \nContext: {context} "
                )
        return prompt

    def qa_chain(self):
        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )
        return chain

    @abstractmethod
    def get_splits(self):
        raise NotImplementedError

    @abstractmethod
    def get_vectorstore(self):
        raise NotImplementedError

    @abstractmethod
    def get_retriever(self):
        raise NotImplementedError