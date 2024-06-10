from chatbots.models.llm import LLMs
from langchain.prompts import PromptTemplate


def get_default_prompt(model_name: LLMs):
    """Get default prompt message based on LLM"""
    if model_name == LLMs.LLAMA2_13B:
        prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you do not know the answer, just say that you don't know. do not try to make up an answer"
            "try to use exact wording from context that is relevant, Keep the answer concise but give details"
            "<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"
        )
    elif model_name == LLMs.GPT_3_PT_5_TURBO:
        prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "Do not try to make up an answer"
            "Please use exact wording from context that is relevant, and give details"
            "\nQuestion: {question} \nContext: {context} "
        )
    else:
        prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. don't try to make up an answer"
            "try to use exact wording from context that is relevant, Keep the answer concise but give details"
            "\nQuestion: {question} \nContext: {context} "
        )
    return prompt
