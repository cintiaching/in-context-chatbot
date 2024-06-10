from abc import ABC, abstractmethod
from typing import List

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from chatbots.models.embedding_models import get_best_embedding_model
from chatbots.models.llm import LLMs, get_llm
from chatbots.utils import load_document
from chatbots.prompt import get_default_prompt
from chatbots.vectorstore.chroma import Vectorstore


class DocumentChatbot(ABC):
    def __init__(self, model_name: LLMs, doc_path: str, chatbot_name: str, vectorstore: Vectorstore = None,
                 prompt_msg: str = None):
        self.model_name = model_name
        self.doc_path = doc_path
        self.prompt_msg = prompt_msg
        self.chatbot_name = chatbot_name

        self.llm = get_llm(model_name)
        self.docs = load_document(doc_path)
        self.prompt = self.get_prompt()
        self.splits = self.get_splits()
        self.vectorstore = vectorstore if vectorstore is not None else self.get_vectorstore()
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
        retriever = self.vectorstore.collection.as_retriever()
        return retriever

    def get_vectorstore(self):
        vectorstore = Vectorstore(
            collection_name=f"{self.chatbot_name}_vectorstore",
            embedding_function=get_best_embedding_model(),
        )
        vectorstore.add_documents(self.splits)
        return vectorstore

    @abstractmethod
    def get_splits(self) -> List[str]:
        raise NotImplementedError
