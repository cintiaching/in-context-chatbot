from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbots.rag.rag import DocumentChatbot


class DocumentBChatbot(DocumentChatbot):
    """
    A Chatbot tailor-made for confidential Document B.
    """
    def get_splits(self) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        all_splits = text_splitter.split_documents(self.docs)
        return [split.page_content for split in all_splits]
