import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

try:
    result = model.invoke('What is the capital of India')
    print(result.content)
except Exception as e:
    print(f"Still failing! Error: {e}")