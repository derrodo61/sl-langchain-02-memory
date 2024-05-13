from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain


load_dotenv()

model = ChatGroq(
    temperature=0.0,
    name="llama3-70b-8192",
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["user_input", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ],
)

chain = LLMChain(llm=model, prompt=prompt, memory=memory)

user_input = "Hello, how are you?"
result = chain.invoke({"user_input": user_input})
print(result["text"])
