import streamlit as st
import uuid
import os
import json
from app.core import get_response

# --- Session/User Setup ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
memory_path = f"memory/{user_id}.json"

# --- Load or Initialize Memory ---
if os.path.exists(memory_path):
    with open(memory_path, "r") as f:
        memory = json.load(f)
else:
    memory = {"chat_history": [], "conversation_context": {}}

# Sync to session state
st.session_state.chat_history = memory.get("chat_history", [])
st.session_state.conversation_context = memory.get("conversation_context", {})

# --- Page Setup ---
st.set_page_config(page_title="CUAB Buddy", layout="centered")
st.title("🎓 CUAB Buddy - Crescent University Assistant")

# --- Chat Display ---
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

# --- Chat Input ---
if user_input := st.chat_input("Ask me anything about Crescent University..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = get_response(user_input)
        placeholder = st.empty()
        full_response = ""
        for part in response.split():
            full_response += part + " "
            placeholder.markdown(full_response + "▌")
            st.sleep(0.03)
        placeholder.markdown(full_response)

    # Save to chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": full_response
    })

    # Save updated memory
    memory = {
        "chat_history": st.session_state.chat_history,
        "conversation_context": st.session_state.conversation_context
    }
    os.makedirs("memory", exist_ok=True)
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)
