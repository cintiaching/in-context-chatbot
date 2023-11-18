import logging
import sys

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

"""
https://medium.com/@bSharpML/use-llamaindex-and-a-local-llm-to-summarize-youtube-videos-29817440e671
"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Change INFO to DEBUG if you want more extensive logging
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = LlamaCPP(
    model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf",

    # optionally, you can set the path to a pre-downloaded model instead of model_url
    # model_path="/Users/cintiaching/Library/Caches/llama_index/models/llama-2-13b-chat.Q5_K_M.gguf",

    temperature=0.0,
    max_new_tokens=1024,

    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=2000, #3900,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.

    # kwargs to pass to __call__()
    generate_kwargs={},

    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 15},

    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

response = llm.complete("What can you tell me about the Ancient Aliens TV Series?")
print(response.text)
