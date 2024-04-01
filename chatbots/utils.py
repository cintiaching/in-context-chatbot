import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFium2Loader

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.environ.get(
    "BASE_DIR",
    ".",
)
LLAMA2_13B_MODEL_PATH = os.path.join(BASE_DIR, "models", "llama-2-13b-chat.Q5_K_M.gguf")


def load_document(file_path):
    """Load document given path, support docx and pdf format"""
    file_extension = file_path.split(".")[-1]
    if file_extension == "docx":
        loader = UnstructuredWordDocumentLoader(
            file_path, strategy="fast",
        )
    elif file_extension == "pdf":
        loader = PyPDFium2Loader(file_path)
    else:
        raise ValueError(f"{file_extension} format is not supported")
    docs = loader.load()
    return docs

