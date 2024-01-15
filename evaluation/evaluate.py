from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLM

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)

"""
https://docs.ragas.io/en/latest/howtos/customisations/azure-openai.html
1. load ground truth
2. load chatbot, run and store the "context" retrieved and answer
3. save data, like model used, hyperparameters, date and time of the run
4. run evaluate
"""


metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
]
azure_model = AzureChatOpenAI(
    deployment_name="your-deployment-name",
    model="your-model-name",
    openai_api_base="https://your-endpoint.openai.azure.com/",
    openai_api_type="azure",
)
# wrapper around azure_model
ragas_azure_model = LangchainLLM(azure_model)
# patch the new RagasLLM instance
answer_relevancy.llm = ragas_azure_model

# init and change the embeddings
# only for answer_relevancy
azure_embeddings = AzureOpenAIEmbeddings(
    deployment="your-embeddings-deployment-name",
    model="your-embeddings-model-name",
    openai_api_base="https://your-endpoint.openai.azure.com/",
    openai_api_type="azure",
)
# embeddings can be used as it is
answer_relevancy.embeddings = azure_embeddings

for m in metrics:
    m.__setattr__("llm", ragas_azure_model)

result = evaluate(
    fiqa_eval["baseline"],
    metrics=metrics,
)

df = result.to_pandas()
print(df.head())