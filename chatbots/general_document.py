from langchain.vectorstores import Chroma
from chatbots.rag import DocumentChatbot
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbots.embedding_choices import EmbeddingModels, EmbeddingConfig, EmbeddingFactory


class GeneralChatbot(DocumentChatbot):
    def __init__(self, doc_path: str, model_name: str, chunk_size=2000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__(doc_path)

    def get_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_splits = text_splitter.split_documents(self.docs)
        return all_splits

    def get_vectorstore(self):
        # using the fastest embedding model for demo
        config = EmbeddingConfig(EmbeddingModels.ALL_MINILM_L12_V2)
        embeddings = EmbeddingFactory.create_embedding(config.model_name)
        vectorstore = Chroma.from_documents(self.splits, embeddings)
        return vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever
