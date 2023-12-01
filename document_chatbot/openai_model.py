import os
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def init_openai_model(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256, *kwarg):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = AzureChatOpenAI(
        callback_manager=callback_manager,
        model_name=model_name,
        deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None),
        temperature=temperature,
        max_tokens=max_tokens,
        *kwarg,
    )
    return llm


class TokenCounter:
    def __init__(self):
        self._count = 0

    def __add__(self, other):
        self._count += other
        return self

    def add(self, other):
        self._count += other

    def total_token_used(self):
        return self._count
