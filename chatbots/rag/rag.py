from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from chatbots.llm.llm import LLMs, get_llm
from chatbots.utils import load_document
from chatbots.prompt import get_default_prompt


class DocumentChatbot(ABC):
    def __init__(self, model_name: LLMs, doc_path: str, collection_name: str, persist_directory: str,
                 prompt_msg: str = None):
        self.model_name = model_name
        self.doc_path = doc_path
        self.prompt_msg = prompt_msg
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.llm = get_llm(model_name)
        self.docs = load_document(doc_path)
        self.prompt = self.get_prompt()
        self.splits = self.get_splits()
        self.vectorstore = self.get_vectorstore()
        self.retriever = self.get_retriever()

    def get_prompt(self):
        if self.prompt_msg:
            prompt = PromptTemplate.from_template(self.prompt_msg)
        else:
            # use default prompt
            prompt = get_default_prompt(self.model_name)
        return prompt

    def qa_chain(self):
        chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )
        return chain

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever

    @abstractmethod
    def get_vectorstore(self):
        raise NotImplementedError

    @abstractmethod
    def get_splits(self):
        raise NotImplementedError


