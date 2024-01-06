from langchain.vectorstores import Chroma
from chatbots.rag import DocumentChatbot
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbots.embedding_choices import get_best_embeddings


class GeneralChatbot(DocumentChatbot):
    def get_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(self.docs)
        return all_splits

    def get_vectorstore(self):
        embeddings = get_best_embeddings()
        vectorstore = Chroma.from_documents(self.splits, embeddings)
        return vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever
