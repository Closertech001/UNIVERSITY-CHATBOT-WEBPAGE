import streamlit as st
import re

def enrich_followup_query(query):
    if "conversation_context" not in st.session_state:
        return query

    ctx = st.session_state.conversation_context
    lower = query.lower()

    # Check if it's a follow-up
    if any(phrase in lower for phrase in ["how about", "what of", "and", "now", "those"]):
        # Inject department if it's missing
        if ctx.get("current_department") and "law" not in lower and "computer" not in lower:
            query = f"{query.strip()} in {ctx['current_department']} department"

        # Inject level if applicable (optional)
        # if ctx.get("current_level") and "100" not in lower:
        #     query += f" for {ctx['current_level']} level"

    return query

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
