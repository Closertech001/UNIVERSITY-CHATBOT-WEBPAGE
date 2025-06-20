import streamlit as st

def detect_emotion(text):
    text = text.lower()
    if any(word in text for word in ["thank", "great", "awesome"]):
        return "positive"
    elif any(word in text for word in ["angry", "frustrated", "upset"]):
        return "negative"
    return "neutral"

def detect_department(query):
    departments = ["computer science", "law"]
    for dept in departments:
        if dept in query:
            return dept
    return None

def update_conversation_context(user_query, response):
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {}
    st.session_state.conversation_context["last_topic"] = user_query
