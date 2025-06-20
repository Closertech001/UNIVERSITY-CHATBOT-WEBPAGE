import streamlit as st
from app.normalizer import DEPARTMENT_ALIASES, normalize_text

def detect_emotion(text):
    text = text.lower()
    if any(word in text for word in ["thank", "great", "awesome", "perfect", "appreciate"]):
        return "positive"
    elif any(word in text for word in ["angry", "frustrated", "upset", "annoyed", "sucks"]):
        return "negative"
    return "neutral"

def detect_department(query):
    query = normalize_text(query)
    for dept, aliases in DEPARTMENT_ALIASES.items():
        if dept in query or any(alias in query for alias in aliases):
            return dept
    return None

def update_conversation_context(user_query, response):
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {}

    query_norm = normalize_text(user_query)

    # Store department context
    for dept, aliases in DEPARTMENT_ALIASES.items():
        if dept in query_norm or any(alias in query_norm for alias in aliases):
            st.session_state.conversation_context["current_department"] = dept
            break

    # Store level context
    for lvl in ["100", "200", "300", "400", "500"]:
        if lvl in query_norm:
            st.session_state.conversation_context["current_level"] = lvl
            break

def enrich_followup_query(query):
    if "conversation_context" not in st.session_state:
        return query

    ctx = st.session_state.conversation_context
    lower = query.lower()

    # Check if user is making a follow-up reference
    followup_phrases = ["how about", "what of", "and", "now", "those"]
    if any(phrase in lower for phrase in followup_phrases):
        # Add department if missing
        if ctx.get("current_department") and ctx["current_department"] not in lower:
            query += f" in {ctx['current_department']} department"
        # Add level if missing
        if ctx.get("current_level") and ctx["current_level"] not in lower:
            query += f" for {ctx['current_level']} level"

    return query
