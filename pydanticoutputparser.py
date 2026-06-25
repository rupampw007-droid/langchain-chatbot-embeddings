from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm= llm)

class Facts(BaseModel):
    Fact1: str = Field(description="Fact 1")
    Fact2: str = Field(description="Fact 2")
    Fact3: str = Field(description="Fact 3")

parser = JsonOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)