from langchain.chains import LLMChain
from langchain.globals import set_debug
from langchain.llms import TextGen
from langchain.prompts import PromptTemplate


model_url = "http://0.0.0.0:1000/"
set_debug(True)

template = """Question: {question}

Answer: Let's think step by step."""


prompt = PromptTemplate(template=template, input_variables=["question"])
llm = TextGen(model_url=model_url)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)
