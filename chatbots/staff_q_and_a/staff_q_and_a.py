from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chatbots.rag import DocumentChatbot


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
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

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
