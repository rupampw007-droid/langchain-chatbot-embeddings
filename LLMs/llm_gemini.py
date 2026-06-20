import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# The library automatically looks for the GOOGLE_API_KEY environment variable
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

try:
    result = model.invoke('What is the capital of India')
    print(result.content)
except Exception as e:
    # This will now catch missing keys or connection issues
    print(f"Error: {e}")