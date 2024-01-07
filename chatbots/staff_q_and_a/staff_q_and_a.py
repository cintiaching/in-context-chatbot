from langchain.vectorstores import Chroma
from chatbots.rag import DocumentChatbot
from chatbots.embedding_choices import EmbeddingModels, EmbeddingConfig, EmbeddingFactory


class StaffQAChatbot(DocumentChatbot):
    def get_splits(self):
        document = self.docs[0].page_content
        result = {}
        question = None
        for text in document.split("\n"):
            if text == "":
                continue
            if "?" in text:
                question = text
                result[question] = []
                continue
            if question is not None:
                result[question].append(text)

        answers = []
        metadatas = []
        for k, v in result.items():
            answers.append("\n".join(v))
            metadatas.append({"question": k})
        return answers, metadatas

    def get_vectorstore(self):
        # using the fastest embedding model for demo
        config = EmbeddingConfig(EmbeddingModels.ALL_MINILM_L12_V2)
        embeddings = EmbeddingFactory.create_embedding(config.model_name)
        vectorstore = Chroma(
            collection_name="q_and_a",
            embedding_function=embeddings,
        )
        answer, metadatas = self.splits
        vectorstore.add_texts(texts=answer, metadatas=metadatas)
        return vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever()
        return retriever
