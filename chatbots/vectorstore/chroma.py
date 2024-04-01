import hashlib
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma


class Vectorstore:
    def __init__(self, collection_name, embedding_function, persist_directory=None, distance_function="cosine"):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.distance_function = distance_function
        # create collection
        self.persist_directory = persist_directory
        if persist_directory is not None:
            setting = Settings(
                is_persistent=True,
                persist_directory=persist_directory,
            )
        else:
            setting = Settings(
                is_persistent=True,
                persist_directory=f"./{self.collection_name}_vectorstore",
            )
        self.collection = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            client=chromadb.Client(setting),
            collection_metadata={"hnsw:space": self.distance_function}
        )

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        if metadatas is not None:
            for _documents, _metadatas in zip(chunk(documents), chunk(metadatas)):
                self.collection.add_texts(
                    texts=_documents,
                    metadatas=_metadatas,
                    ids=get_document_ids(_documents),  # items with existing IDs will not be inserted
                )
        else:
            for _documents in chunk(documents):
                self.collection.add_texts(
                    texts=_documents,
                    ids=get_document_ids(_documents),
                )


def chunk(lst, batch_size=100):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def get_document_ids(documents: List[str]) -> List[str]:
    """
    Generates a list of document IDs based on the SHA-256 hash of each document.
    :param documents: A list of document strings
    :return: A list of document IDs
    """
    return [hashlib.sha256(doc.encode()).hexdigest() for doc in documents]
