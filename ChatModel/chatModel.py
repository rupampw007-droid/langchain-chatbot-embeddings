import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Token loaded:", token[:10] if token else "NOT FOUND")  # debug check

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is Narendra Modi")
print(result.content)