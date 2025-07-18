import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv
import time
import re

# Load env vars
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Dictionaries for synonyms and abbreviations ---
SYNONYMS = {
    "program": "course",
    "courses": "subjects",
    "exam": "examination",
    "lecturer": "instructor",
    "study": "learn",
    "major": "department",
    "dorm": "hostel",
}

ABBREVIATIONS = {
    "vc": "vice chancellor",
    "hod": "head of department",
    "cgpa": "cumulative grade point average",
    "gp": "grade point",
    "gpa": "grade point average",
    "dept": "department",
    "uni": "university",
    "cuab": "crescent university",
}

# --- Normalize query ---
def normalize_query(text):
    # Lowercase and expand abbreviations
    for abbr, full in ABBREVIATIONS.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)

    # Replace synonyms
    for syn, std in SYNONYMS.items():
        pattern = r"\b" + re.escape(syn) + r"\b"
        text = re.sub(pattern, std, text, flags=re.IGNORECASE)

    # Correct typos
    text = correct_text(text)
    return text

# --- Correct typos ---
def correct_text(text):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- Search with FAISS ---
def search(query, index, model, chunks, top_k=1):
    query_emb = model.encode(query).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    match = chunks[I[0][0]]
    score = D[0][0]
    return match, score

# --- Ask GPT-4 with fallback ---
def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful Crescent University assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"], "gpt-4"
    except Exception as e:
        return "Sorry, I'm having trouble reaching GPT-4 right now. Please try again shortly.", "fallback"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🎓 Crescent Uni Assistant", layout="wide")
st.title("🎓 Crescent University Chatbot")
st.caption("Ask me anything about Crescent University!")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dict_path = "frequency_dictionary_en_82_765.txt"
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

@st.cache_resource
def load_chunks_and_index():
    with open("data/qa_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    dim = 384
    index = faiss.IndexFlatL2(dim)
    embeddings = [model.encode(chunk["question"] + " " + chunk["answer"]) for chunk in chunks]
    index.add(np.array(embeddings).astype("float32"))

    return chunks, index, embeddings

# Load all resources
model = load_model()
symspell = load_symspell()
chunks, index, embeddings = load_chunks_and_index()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("You:", key="input")

if user_input:
    norm_query = normalize_query(user_input)
    match, score = search(norm_query, index, model, chunks)

    threshold = 0.4
    if score < threshold:
        response, source = ask_gpt(user_input)
    else:
        response = match["answer"]
        source = "dataset"

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append((source, response))

# Display chat
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"**You:** {msg}")
    elif sender == "gpt-4":
        st.markdown(f"**CrescentBot (GPT-4):** {msg}")
    elif sender == "fallback":
        st.markdown(f"**CrescentBot:** {msg}")
    else:
        st.markdown(f"**CrescentBot (Local Data):** {msg}")
