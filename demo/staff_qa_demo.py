from chatbots.rag.staff_q_and_a import StaffQAChatbot
from chatbots.models.llm import LLMs

print(f"""Model Choices: 1) {LLMs.GPT_3_PT_5_TURBO.value} 2) {LLMs.LLAMA2_13B.value} 3) {LLMs.MISTRAL_7B.value}""")
choice = input(f"Select model: ")

if choice == "1":
    selected_model = LLMs.GPT_3_PT_5_TURBO.value
elif choice == "2":
    selected_model = LLMs.LLAMA2_13B.value
elif choice == "3":
    selected_model = LLMs.MISTRAL_7B.value
else:
    raise ValueError(f"Please select a valid model.")

staff_qa_chatbot = StaffQAChatbot(
    doc_path="data/New Staff Handbook Q&A.docx",
    model_name=LLMs(selected_model),
    collection_name="staff_qa",
    persist_directory="./data/staff_qa_vectorstore"
)
chain = staff_qa_chatbot.qa_chain()

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

for question in jacks_questions:
    print(f"Question: {question}\n")
    print(f"Answer: {chain.run(question)}\n")
