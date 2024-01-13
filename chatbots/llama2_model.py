from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

from chatbots.utils import LLAMA2_13B_MODEL_PATH


def init_llama2_13b_llm(
        model_path=None,
        n_gpu_layers=1000,
        n_ctx=2048,
        verbose=False,
        *kwarg
):
    if model_path is None:
        model_path = LLAMA2_13B_MODEL_PATH
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # if the model is not stored locally, download from
    # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,  # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
        n_ctx=n_ctx,  # Context size, text limits for responses
        # f16_kv MUST set to True, otherwise you will run into problem after a couple of calls
        # Use half-precision for key/value cache.
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=verbose,
        *kwarg
    )
    return llm
