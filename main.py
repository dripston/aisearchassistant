import streamlit as st
import logging
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Backend setup
# -----------------------------------------------------------------------------
search = DuckDuckGoSearchRun()
llm = ChatOllama(model="llama3.1", temperature=0.1)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def truncate_search_results(search_result: str, max_chars: int = 1500) -> str:
    """Truncate search results to prevent overwhelming the LLM"""
    if len(search_result) <= max_chars:
        return search_result
    
    # Try to truncate at a sentence boundary
    truncated = search_result[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.7:  # If we can find a period in the last 30%
        return truncated[:last_period + 1]
    else:
        return truncated + "..."

def extract_key_info(search_result: str) -> str:
    """Extract and summarize key information from search results"""
    # Split into sentences and take the most relevant ones
    sentences = [s.strip() for s in search_result.split('.') if len(s.strip()) > 20]
    
    # Take first few sentences that are likely to contain key info
    key_sentences = sentences[:5]  # Limit to 5 sentences max
    
    return '. '.join(key_sentences) + '.' if key_sentences else search_result[:500]

graph_builder = StateGraph(State)

def chatbot(state: State):
    log.info("Processing user request...")
    
    if not state["messages"]:
        ai_msg = AIMessage(content="Hi! Ask me anything and I'll search for the latest information.")
        return {"messages": [ai_msg]}

    last_message = state["messages"][-1]

    # Extract user question
    if hasattr(last_message, "content"):
        user_question = last_message.content
    elif isinstance(last_message, dict):
        user_question = last_message.get("content", "")
    else:
        user_question = str(last_message)

    log.info(f"User question: {user_question[:100]}...")

    # --- Optimized Search ---
    try:
        log.info("Searching...")
        raw_search_result = search.invoke(user_question)
        
        # Truncate and optimize search results
        optimized_search_result = extract_key_info(raw_search_result)
        log.info(f"Search completed, processed {len(raw_search_result)} -> {len(optimized_search_result)} chars")
        
    except Exception as e:
        log.error(f"Search failed: {e}")
        optimized_search_result = "Search temporarily unavailable."

    # --- Optimized System Prompt ---
    system_prompt = f"""Based on the search results below, provide a concise, well-structured answer to the user's question.

SEARCH RESULTS:
{optimized_search_result}

USER QUESTION: {user_question}

Instructions:
- Keep response under 300 words
- Focus on the most recent and relevant information
- Use clear, structured formatting
- If search results are limited, acknowledge this
"""

    # --- LLM with better error handling ---
    try:
        log.info("Generating response...")
        lc_response = llm.invoke(system_prompt)
        content = getattr(lc_response, "content", str(lc_response))
        log.info("Response generated successfully")
        
    except Exception as e:
        log.error(f"LLM failed: {e}")
        content = f"I encountered an error generating the response. Please try rephrasing your question."

    ai_message = AIMessage(content=content)
    return {"messages": state["messages"] + [ai_message]}

# Wire graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# -----------------------------------------------------------------------------
# Streamlit frontend with performance optimizations
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fast Research Chat", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("⚡ Fast Research Chat")
st.caption("Optimized for speed and efficiency")

# Initialize session state
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []

# Display messages with better formatting
for msg in st.session_state.lc_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input with processing indicator
user_input = st.chat_input("Ask your question (I'll search for the latest info)")

if user_input:
    # Add user message immediately
    st.session_state.lc_messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show processing indicator
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and analyzing..."):
            try:
                # Invoke graph
                new_state = graph.invoke({"messages": st.session_state.lc_messages})
                st.session_state.lc_messages = new_state["messages"]
                ai_reply = st.session_state.lc_messages[-1].content
                
                # Display response
                st.markdown(ai_reply)
                
            except Exception as e:
                log.error(f"Graph invocation failed: {e}")
                error_msg = "I'm experiencing technical difficulties. Please try again in a moment."
                st.error(error_msg)
                st.session_state.lc_messages.append(AIMessage(content=error_msg))

# Add sidebar with performance info
with st.sidebar:
    st.header("Performance Info")
    st.write(f"Messages in conversation: {len(st.session_state.lc_messages)}")
    
    if st.button("Clear Conversation"):
        st.session_state.lc_messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Optimizations applied:")
    st.caption("• Truncated search results")
    st.caption("• Reduced LLM prompt size") 
    st.caption("• Limited response length")
    st.caption("• Improved error handling")