from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from document_chatbot.rag import DocumentChatbot
from langchain.text_splitter import RecursiveCharacterTextSplitter


class StaffHandbookChatbot(DocumentChatbot):
    def get_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        all_splits = text_splitter.split_documents(self.docs)
        metadatas = None
        return all_splits, metadatas

    def get_vectorstore(self):
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"    # To be changed
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        vectorstore = Chroma(
            collection_name="staff_handbook",
            embedding_function=embeddings,
        )
        answer, metadatas = self.splits
        vectorstore.add_texts(texts=answer, metadatas=metadatas)
        return vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever
