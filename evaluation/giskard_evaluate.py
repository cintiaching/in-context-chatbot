import giskard
import pandas as pd

from document_chatbot.staff_q_and_a.staff_q_and_a import StaffQAChatbot

# refer to https://docs.giskard.ai/en/stable/getting_started/quickstart/quickstart_llm.html

staff_qa_chatbot = StaffQAChatbot(
    doc_path="data/New Staff Handbook Q&A.docx",
    model_name="llama2_13b",
)
qa_chain = staff_qa_chatbot.qa_chain()


def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [qa_chain.run({"query": question}) for question in df["question"]]


# Don’t forget to fill the `name` and `description`: they are used by Giskard
# to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Staff Handbook Question Answering",
    description="This model answers any question about rules about a company based on a Staff Handbook FAQ",
    feature_names=["question"],
)

# Optional: let’s test that the wrapped model works
jacks_questions = [
    "if I forgot my password, how can I accept the handbook?",
    "where I can find the latset handbook and get to know if I acknowledged the latest version",
    "I am not sure if I have the username and password to the portal, maybe I and the first time using the portal",
    "if Dental services covered by our insurance plan? and what is the coverage?",
    "if I disagree with the handbook terms, what can I do?",
    "do I need to acknowledge the handbook? as I already acknowledged during my onboarding process few years ago",
    "can I wear sandle back to office",
    "can I wear smart casual back to office",
    "if our company insurance covers OT period?",
    "if our company insurance covers staff uniform? if I lost, do I need to pay for that",
    "do I need to dress uniform for work",
    "if I die, if company compensiate my family",
]

giskard_dataset = giskard.Dataset(pd.DataFrame({"question": jacks_questions}), target=None)
print(giskard_model.predict(giskard_dataset).prediction)

report = giskard.scan(giskard_model, giskard_dataset, only="hallucination")
print(report)

full_report = giskard.scan(giskard_model, giskard_dataset)
print(full_report)

full_report.to_html("test_report.html")
