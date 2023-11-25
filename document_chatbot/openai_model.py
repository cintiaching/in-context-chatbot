from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def init_openai_model(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256, *kwarg):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = AzureChatOpenAI(
        callback_manager=callback_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        *kwarg,
    )
    return llm
