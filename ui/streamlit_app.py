import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from app.core import get_response

st.set_page_config(page_title="CUAB Buddy", layout="centered")
st.title("🎓 CUAB Buddy - Crescent University Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        response = get_response(user_input)
    full_response = response
    st.session_state.chat_history.append({"user": user_input, "bot": full_response})
