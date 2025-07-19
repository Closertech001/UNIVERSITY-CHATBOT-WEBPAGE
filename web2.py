# web.py - Crescent University Chatbot (Enhanced)

import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
import re
import time

# --- Load Environment Variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Synonyms & Abbreviations ---
SYNONYMS = {
    "program": "course", "courses": "subjects", "exam": "examination",
    "lecturer": "instructor", "study": "learn", "major": "department",
    "dorm": "hostel"
}
ABBREVIATIONS = {
    "vc": "vice chancellor", "hod": "head of department",
    "cgpa": "cumulative grade point average", "gp": "grade point",
    "gpa": "grade point average", "dept": "department",
    "uni": "university", "cuab": "crescent university"
}

# --- Normalize and Correct Input ---
def normalize_query(text):
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(r"\\b" + re.escape(abbr) + r"\\b", full, text, flags=re.IGNORECASE)
    for syn, std in SYNONYMS.items():
        text = re.sub(r"\\b" + re.escape(syn) + r"\\b", std, text, flags=re.IGNORECASE)
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- FAISS Semantic Search ---
def search(query, index, model, questions, top_k=1):
    query_emb = model.encode(query).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    return I[0][0], D[0][0]

# --- GPT-4 Fallback ---
def ask_gpt(prompt, history=None):
    try:
        chat_log = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in (history or [])[-3:]])
        full_prompt = (
            "You are a knowledgeable and friendly assistant for Crescent University. "
            "Answer only based on information relevant to the university. If you are unsure, say 'I don't know.'\n\n"
            f"Chat History:\n{chat_log}\n\nUser: {prompt}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful Crescent University chatbot."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.4, max_tokens=600
        )
        return response["choices"][0]["message"]["content"], "gpt-4"
    except Exception:
        return "Sorry, I'm currently unable to fetch a response from GPT-4.", "fallback"

# --- Greeting Detection ---
def detect_greeting(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(g in user_input.lower() for g in greetings)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🎓 Crescent Uni Assistant", layout="wide")
st.title("🎓 Crescent University Chatbot")
st.caption("Ask me anything about courses, departments, staff, or university life!")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_symspell():
    sym = SymSpell(max_dictionary_edit_distance=2)
    sym.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    return sym

@st.cache_resource
def load_chunks_index():
    with open("data/qa_data_cleaned.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    questions = [q["question"] for q in chunks]
    embeddings = model.encode(questions, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index, questions

# --- Load Resources ---
model = load_model()
symspell = load_symspell()
chunks, index, questions = load_chunks_index()

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat UI ---
user_input = st.text_input("You:", key="input")
if user_input:
    if detect_greeting(user_input):
        response = "Hi there! How can I assist you today at Crescent University?"
        source = "greeting"
    else:
        norm_query = normalize_query(user_input)
        idx, score = search(norm_query, index, model, questions)
        threshold = 0.65

        if score < threshold:
            response, source = ask_gpt(user_input, st.session_state.chat_history)
        else:
            response, source = chunks[idx]["answer"], "dataset"

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append((source, response))

# --- Display History ---
for sender, msg in st.session_state.chat_history:
    label = {
        "user": "**You:**",
        "dataset": "**CrescentBot (Local):**",
        "gpt-4": "**CrescentBot (GPT-4):**",
        "fallback": "**CrescentBot:**",
        "greeting": "**CrescentBot:**"
    }.get(sender, "**CrescentBot:**")
    st.markdown(f"{label} {msg}")
