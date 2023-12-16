import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.environ.get(
    "BASE_DIR",
    ".",
)
LLAMA2_13B_MODEL_PATH = os.path.join(BASE_DIR, "models", "llama-2-13b-chat.Q5_K_M.gguf")
