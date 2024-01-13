from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.errors import InvalidDimensionException

from document_chatbot.rag import DocumentChatbot


class StaffHandbookChatbot(DocumentChatbot):
    def get_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        all_splits = text_splitter.split_documents(self.docs)
        return all_splits

    def get_vectorstore(self):
        # using the fastest embedding model for demo
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        try:
            vectorstore = Chroma.from_documents(self.splits, embeddings)
        except InvalidDimensionException:
            Chroma().delete_collection()
            vectorstore = Chroma.from_documents(self.splits, embeddings)
        return vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever
