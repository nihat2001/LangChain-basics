from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(api_key=os.getenv("API_KEY"),
               model="llama-3.3-70b-versatile", 
               temperature=0.5,
               max_tokens=500)

parser = StrOutputParser()

# ---------------BASIC METHOD------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "What is {topic}?" )
])

chain = prompt | llm | parser

result = chain.invoke({"topic": "Machine Learning"})
print(f"Result: {result}")

# ---------------RAG METHOD--------------------
def get_context(q:str) -> str: return "Baku is the capital of Azerbaijan with 3M+ population"

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context:\n{context}"),
    ("human", "{question}")
])

rag_chain = (
        RunnablePassthrough.assign(context = lambda x: get_context(x["question"]))
        | rag_prompt | llm | parser
)

rag_result = rag_chain.invoke({"question": "What can you tell me about Baku?"})
print(f"RAG Result: {rag_result}")

# ---------------LAMBDA METHOD-------------------
def get_upper(text: str) -> str:
    return text.upper()

get_upper_function = RunnableLambda(get_upper)

lambda_chain = prompt | llm | parser | get_upper_function

lambda_result = lambda_chain.invoke({"topic": "Deep Learning"})
print(f"Lambda Result: {lambda_result}")

# ---------------PARALLEL METHOD-------------------
simple_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {topic} as simply?")
])   | llm | parser 

technical_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {topic} in technical terms?")
])   | llm | parser

paralel_chain = RunnableParallel(simple = simple_prompt, technical = technical_prompt).invoke({"topic": "Artificial Intelligence"})
print(f"Simple Result: {paralel_chain['simple']}")
print(f"Technical Result: {paralel_chain['technical']}")

# ---------------HISTORY METHOD-------------------
memory = {}

def get_history(sid):
    memory.setdefault(sid, InMemoryChatMessageHistory())
    return memory[sid]

memory_chain = RunnableWithMessageHistory(
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{history}"),
        ("human", "{input}")
    ])  | llm | parser, 
    get_history,
    input_messae_key="input",
    history_message_key="history"
)

config = {"configurable": {"session_id": "user1"}}
print(memory_chain.invoke({"input": "My name is Nihat."}, config=config))
print(memory_chain.invoke({"input": "What is my name?"}, config=config))
