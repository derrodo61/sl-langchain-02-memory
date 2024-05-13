# from dotenv import load_dotenv
import streamlit as st
from pydantic.v1.types import SecretStr

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain


st.title("RD - Chat with non-persistent memory")

# sidebar
groq_api_key = SecretStr(st.sidebar.text_input("GROQ_API_KEY"))
submit_button = st.sidebar.button("Submit")

# memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# prompt
prompt = ChatPromptTemplate(
    input_variables=["user_input", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ],
)


def generate_response(user_input):
    model = ChatGroq(
        temperature=0.0,
        name="llama3-70b-8192",
        api_key=groq_api_key,
    )
    chain = LLMChain(llm=model, prompt=prompt, memory=memory, verbose=True)
    result = chain.invoke({"user_input": user_input})
    st.info(result["text"])
    st.info(result)


with st.form("my_form"):
    text = st.text_area(
        "Enter your message here",
        value="Hello, how are you?",
    )
    submitted = st.form_submit_button("Submit")
    if not groq_api_key.get_secret_value().startswith("gsk"):
        st.warning("Please enter your Groq API key!", icon="⚠")
    if submitted and groq_api_key.get_secret_value().startswith("gsk"):
        generate_response(text)
