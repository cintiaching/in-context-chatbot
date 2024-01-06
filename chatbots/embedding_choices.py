from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


def get_best_embeddings():
    """Highest rank in MTEB English leaderboard (https://huggingface.co/spaces/mteb/leaderboard) and implemented here"""
    return bge_large_en_v1_5()


def bge_large_en_v1_5():
    """Rank 5 in MTEB English leaderboard on 6 Jan 2024 (https://huggingface.co/spaces/mteb/leaderboard)"""
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings


def all_MiniLM_L6_v2():
    """fast and simple, rank 53"""
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    return embeddings


def all_MiniLM_L12_v2():
    """fast and simple, rank 52"""
    embeddings_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    return embeddings


def instruct_embedding(query_instruction="Represent the query for retrieval: "):
    """instruct embedding models"""
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction=query_instruction,
    )
    return embeddings
