# --- Streamlit University Chatbot Template ---

import streamlit as st
import json
import re
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import openai
import os

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]  # or use os.environ["OPENAI_API_KEY"]

# --- Load Spell Correction ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# --- Load Semantic Model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Load Q&A Data ---
with open("qa_data.json", "r") as f:
    qa_data = json.load(f)
    questions = [item['question'] for item in qa_data]
    answers = [item['answer'] for item in qa_data]
    embeddings = model.encode(questions, convert_to_tensor=True)

# --- Abbreviation Normalization ---
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
    prompt = f"You are a helpful university assistant. Answer the following question clearly and concisely based only on university-related context:\n\nQuestion: {user_input}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent Uni Assistant", layout="wide")
st.title("🎓 Crescent University Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.text_input("Ask me anything about Crescent University:", key="user_input")

if st.button("Ask") and user_query:
    processed = preprocess_input(user_query)
    score, best_answer, matched_question = get_best_match(processed)

    if score > 0.70:
        answer = best_answer
    else:
        answer = fallback_response(user_query)

    st.session_state.history.append((user_query, answer))

for q, a in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
