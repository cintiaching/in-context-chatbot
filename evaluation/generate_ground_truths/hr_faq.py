import os
import pandas as pd
from langchain.document_loaders import UnstructuredWordDocumentLoader


def get_splits(docs):
    document = docs[0].page_content
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


doc_path = "data/New Staff Handbook Q&A.docx"
loader = UnstructuredWordDocumentLoader(
    doc_path, strategy="fast",
)
docs = loader.load()
gt, questions = get_splits(docs)
result = pd.DataFrame({"question": questions, "ground_truths": gt})

output_path = ".../data/ground_truths/"
os.makedirs(output_path, exist_ok=True)
result.to_csv(os.path.join(output_path, "hr_faq.csv"), index=False)
