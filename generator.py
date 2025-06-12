from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load token from .env
load_dotenv()

# Initialize Hugging Face API
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set up LLM endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512,
    temperature=0.7,
)

# Wrap in chat-friendly interface
chat_model = ChatHuggingFace(llm=llm_endpoint)

# Flashcard prompt template
prompt_template = PromptTemplate(
    input_variables=["subject", "content"],
    template="""
Generate 10 to 15 flashcards in Q&A format from the following educational content.

Subject: {subject}

Content:
\"\"\"
{content}
\"\"\"

Use this format only:
Q: [question]
A: [answer]

Return only the flashcards.
"""
)

# Flashcard generation function
def generate_flashcards(content, subject="General"):
    prompt = prompt_template.format(subject=subject, content=content)
    result = chat_model.invoke(prompt).content

    flashcards = []
    question, answer = "", ""
    for line in result.split('\n'):
        if line.strip().startswith("Q:"):
            question = line.strip()[2:].strip()
        elif line.strip().startswith("A:"):
            answer = line.strip()[2:].strip()
            if question and answer:
                flashcards.append({"question": question, "answer": answer})
                question, answer = "", ""
    return flashcards
