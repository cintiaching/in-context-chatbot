from chatbots.rag.rag import DocumentChatbot
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbots.models.llm import LLMs


class GeneralChatbot(DocumentChatbot):
    def __init__(self, doc_path: str, model_name: LLMs, collection_name, persist_directory, chunk_size=2000,
                 chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__(model_name, doc_path, collection_name, persist_directory)

    def get_splits(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_splits = text_splitter.split_documents(self.docs)
        return [split.page_content for split in all_splits]
