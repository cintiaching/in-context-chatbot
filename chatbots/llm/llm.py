import os
from enum import Enum
from dotenv import load_dotenv

from langchain.llms import LlamaCpp
from langchain.llms import Ollama
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from chatbots.utils import LLAMA2_13B_MODEL_PATH

# Load environment variables from .env file
load_dotenv()


class LLMs(Enum):
    LLAMA2_13B = "llama-2-13b-chat.Q5_K_M"
    GPT_3_PT_5_TURBO = "gpt-3.5-turbo"
    MISTRAL_7B = "mistral"


class LLMConfig:
    def __init__(self, model_name, model_path=None, **kwarg):
        self.model_name = model_name
        self.model_path = model_path
        self.model_kwarg = kwarg


class LLMFactory:
    @staticmethod
    def initiate_llm(config: LLMConfig):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # Add more cases if needed for different llms
        if config.model_name == LLMs.LLAMA2_13B:
            if config.model_path is None:
                config.model_path = LLAMA2_13B_MODEL_PATH
            if config.model_kwarg is None:
                # use default config
                config.model_kwarg = {
                    "n_gpu_layers": 1000,  # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
                    "n_ctx": 2048,  # Context size, text limits for responses
                    "verbose": False,
                }
            return LlamaCpp(
                model_path=config.model_path,
                f16_kv=True,  # f16_kv MUST set to True, issue with llamacpp
                callback_manager=callback_manager,
                **config.model_kwarg,
            )
        elif config.model_name == LLMs.GPT_3_PT_5_TURBO:
            if config.model_kwarg is None:
                # use default config
                config.model_kwarg = {
                    "temperature": 0,
                    "max_tokens": 256,
                }
            return AzureChatOpenAI(
                model_name=config.model_name.value,
                callback_manager=callback_manager,
                deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None),
                **config.model_kwarg,
            )
        elif config.model_name == LLMs.MISTRAL_7B:
            return Ollama(
                model=config.model_name.value,
                callback_manager=callback_manager,
                **config.model_kwarg,
            )
        else:
            ValueError(f"Invalid LLM config with model name '{config.model_name}'")


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
