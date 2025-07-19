# web.py - Crescent University Chatbot (Cleaned Q&A + FAISS + GPT-4 Fallback)

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
def search(query, index, model, chunks, top_k=1):
    query_emb = model.encode(query).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    return chunks[I[0][0]], D[0][0]

# --- GPT-4 Fallback ---
def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful Crescent University assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, max_tokens=600
        )
        return response["choices"][0]["message"]["content"], "gpt-4"
    except Exception:
        return "Sorry, I'm currently unable to fetch a response from GPT-4.", "fallback"

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
    texts = [q["question"] + " " + q["answer"] for q in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index

# --- Load Resources ---
model = load_model()
symspell = load_symspell()
chunks, index = load_chunks_index()

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat UI ---
user_input = st.text_input("You:", key="input")
if user_input:
    norm_query = normalize_query(user_input)
    match, score = search(norm_query, index, model, chunks)
    threshold = 0.45

    if score < threshold:
        response, source = ask_gpt(user_input)
    else:
        response, source = match["answer"], "dataset"

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append((source, response))

# --- Display History ---
for sender, msg in st.session_state.chat_history:
    label = {
        "user": "**You:**",
        "dataset": "**CrescentBot:**",
        "gpt-4": "**CrescentBot:**",
        "fallback": "**CrescentBot:**"
    }.get(sender, "**CrescentBot:**")
    st.markdown(f"{label} {msg}")
