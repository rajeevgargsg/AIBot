import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Groq Agent", layout="wide")
st.title("⚡The Tool Toggle")

with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key (Optional)", type="password")
    use_search = st.toggle("Enable Web Search", value=True)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tools Setup
os.environ["TAVILY_API_KEY"] = tavily_key if tavily_key else "dummy_key"
web_tool = TavilySearchResults(max_results=2)
active_tools = [web_tool] if (use_search and tavily_key) else []

# UI: Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Logic: Chat Input
if user_input := st.chat_input():
    if not groq_key:
        st.error("Please enter your Groq API Key in the sidebar!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=groq_key)
    agent_executor = create_react_agent(llm, tools=active_tools)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            inputs = {"messages": [HumanMessage(content=user_input)]}
            response = agent_executor.invoke(inputs)
            final_answer = response["messages"][-1].content
            st.write(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
