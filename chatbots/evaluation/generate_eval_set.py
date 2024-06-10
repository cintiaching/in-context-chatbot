import os

import pandas as pd
from datasets import Dataset

from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper

from chatbots.models.llm import LLMs, LLMConfig, LLMFactory
from chatbots.models.embedding_models import EmbeddingModels, EmbeddingConfig, EmbeddingFactory


def get_staff_q_and_a():
    """Get questions and answers specifically from Staff Handbook Q&A.docx"""
    # load document
    loader = UnstructuredWordDocumentLoader(
        "../data/New Staff Handbook Q&A.docx", strategy="fast",
    )
    document = loader.load()
    document = document[0].page_content
    result = {}
    question = None
    for text in document.split("\n"):
        if text == "":
            continue
        if "?" in text:
            question = text
            result[question] = []
            continue
        if question is not None:
            result[question].append(text)

    answers = []
    questions = []
    for k, v in result.items():
        answers.append("\n".join(v))
        questions.append(k)
    return answers, questions


if __name__ == "__main__":
    answers, questions = get_staff_q_and_a()
    # use the answer as context and ground truths
    eval_dataset = []
    for a, q in zip(answers, questions):
        d = {"question": q, "answer": a, "contexts": [a], "ground_truth": a}
        eval_dataset.append(d)
    eval_dataset_df = pd.DataFrame(eval_dataset)

    # test eval using ragas
    # define metrics
    metrics = [
        answer_relevancy,
        context_precision,
        faithfulness,
        context_recall,
    ]
    # set up the llm and embedding model for evaluation
    config = LLMConfig(model_name=LLMs.MISTRAL_7B)
    llm_wrapper = LangchainLLMWrapper(LLMFactory.initiate_llm(config))
    config = EmbeddingConfig(EmbeddingModels.ALL_MINILM_L12_V2)
    embeddings = EmbeddingFactory.create_embedding(config.model_name, config.model_kwargs, config.encode_kwargs)
    for m in metrics:
        m.__setattr__("llm", llm_wrapper)
        if hasattr(m, "embeddings"):
            m.__setattr__("embeddings", embeddings)

    # make the DatasetDict
    eval_dataset = Dataset.from_pandas(eval_dataset_df)
    # evaluate
    result = evaluate(
        eval_dataset,
        metrics=metrics,
    )
    df = result.to_pandas()
    os.makedirs("../artifact/", exist_ok=True)
    df.to_csv("../artifact/test_eval.csv", index=False)
