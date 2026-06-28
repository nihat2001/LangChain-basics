
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(api_key=os.getenv("API_KEY"),
               model="llama-3.3-70b-versatile", 
               temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator. Translate the user's input into {language}."),
    ("user", "{text}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"language": "Azerbaijani", "text": "Hello, how are you?"})

print(f"Result: {result}") 
