from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.errors import InvalidDimensionException

from chatbots.rag.rag import DocumentChatbot
from chatbots.llm.embedding_models import EmbeddingModels, EmbeddingConfig, EmbeddingFactory


class StaffHandbookChatbot(DocumentChatbot):
    def get_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        all_splits = text_splitter.split_documents(self.docs)
        return all_splits

    def get_vectorstore(self):
        # using the fastest embedding model for demo
        config = EmbeddingConfig(EmbeddingModels.ALL_MINILM_L12_V2)
        embeddings = EmbeddingFactory.create_embedding(config.model_name, config.model_kwargs, config.encode_kwargs)
        try:
            vectorstore = Chroma.from_documents(self.splits, embeddings, collection_name=self.collection_name,
                                                persist_directory=self.persist_directory)
        except InvalidDimensionException:
            Chroma().delete_collection()
            vectorstore = Chroma.from_documents(self.splits, embeddings, collection_name=self.collection_name,
                                                persist_directory=self.persist_directory)
        return vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever
