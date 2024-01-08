from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from enum import Enum


class EmbeddingModels(Enum):
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    ALL_MINILM_L12_V2 = "sentence-transformers/all-MiniLM-L12-v2"
    INSTRUCT_EMBEDDING = "hkunlp/instructor-large"


class EmbeddingConfig:
    def __init__(self, model_name, model_kwargs=None, encode_kwargs=None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs


class EmbeddingFactory:
    @staticmethod
    def create_embedding(model_name, model_kwargs=None, encode_kwargs=None, **kwargs):
        # Add more cases if needed for different embedding models
        if model_name == EmbeddingModels.BGE_LARGE_EN_V1_5:
            return HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        elif model_name == EmbeddingModels.ALL_MINILM_L6_V2:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif model_name == EmbeddingModels.ALL_MINILM_L12_V2:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif model_name == EmbeddingModels.INSTRUCT_EMBEDDING:
            return HuggingFaceInstructEmbeddings(
                query_instruction=kwargs.get("query_instruction", "Represent the query for retrieval: "),
            )
        else:
            raise ValueError("Invalid embedding model name")


def get_best_embedding_model():
    """Returns the best embedding model based on the MTEB leaderboard."""
    model_name = EmbeddingModels.BGE_LARGE_EN_V1_5.value
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    config = EmbeddingConfig(model_name, model_kwargs, encode_kwargs)
    embeddings = EmbeddingFactory.create_embedding(
        config.model_name,
        config.model_kwargs,
        config.encode_kwargs,
    )
    return embeddings
