# --- Streamlit University Chatbot with Enhanced Memory and UI ---

import streamlit as st
import json
import re
import time
import sqlite3
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import openai
import os

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- SQLite Setup for Long-Term Memory ---
conn = sqlite3.connect("chat_memory.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS memory (
    session_id TEXT,
    user_name TEXT,
    user_dept TEXT,
    question TEXT,
    answer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# --- Spell Correction ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# --- Semantic Model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Q&A Data ---
with open("qa_data.json", "r") as f:
    qa_data = json.load(f)
    questions = [item['question'] for item in qa_data]
    answers = [item['answer'] for item in qa_data]
    embeddings = model.encode(questions, convert_to_tensor=True)

# --- Normalization ---
ABBREVIATIONS = {
    "u": "you", "ur": "your", "r": "are", "dept": "department",
    "info": "information", "sch": "school", "cn": "can"
}

def normalize_text(text):
    words = text.split()
    return ' '.join([ABBREVIATIONS.get(w.lower(), w) for w in words])

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def preprocess_input(user_input):
    text = normalize_text(user_input)
    return correct_spelling(text)

def get_best_match(user_input):
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_score = float(scores.max())
    best_idx = int(scores.argmax())
    return best_score, answers[best_idx], questions[best_idx]

def fallback_response(user_input):
    name = st.session_state.get("user_name", "a student")
    dept = st.session_state.get("user_dept", "the university")
    prompt = f"You are a helpful assistant for Crescent University. You are talking to {name} from {dept}. Answer the following question based only on university-related context.\n\nQuestion: {user_input}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"].strip()

# --- UI Setup ---
st.set_page_config(page_title="🎓 Crescent Uni Assistant", layout="wide")

st.markdown("""
    <style>
    .message-bubble-user {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: right;
    }
    .message-bubble-bot {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .bot-name {
        font-weight: bold;
        color: #4b4b4b;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")
st.markdown("👋 Welcome! Ask me anything about **admissions**, **courses**, **fees**, **departments**, or **campus life**.")

# --- Memory Setup ---
if "user_name" not in st.session_state:
    st.session_state.user_name = st.text_input("👤 What's your name?")

if "user_dept" not in st.session_state:
    st.session_state.user_dept = st.text_input("🏫 What department are you in?")

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.user_name and st.session_state.user_dept:
    user_query = st.text_input("💬 Your question:", key="user_input")
    if st.button("Ask") and user_query:
        with st.spinner("Typing..."):
            processed = preprocess_input(user_query)
            score, best_answer, matched_question = get_best_match(processed)

            if score > 0.70:
                answer = best_answer
            else:
                answer = fallback_response(user_query)

            st.session_state.history.append((user_query, answer))

            # Store to DB
            c.execute("""
                INSERT INTO memory (session_id, user_name, user_dept, question, answer)
                VALUES (?, ?, ?, ?, ?)
            """, (st.session_state.user_name, st.session_state.user_name, st.session_state.user_dept, user_query, answer))
            conn.commit()
else:
    st.info("Please fill in your name and department to start chatting.")

# --- Display Chat History ---
for q, a in reversed(st.session_state.history):
    st.markdown(f"<div class='message-bubble-user'>🙋‍♂️ {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='message-bubble-bot'><span class='bot-name'>🤖 Bot:</span> {a}</div>", unsafe_allow_html=True)
