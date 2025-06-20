import json
from sentence_transformers import SentenceTransformer
import streamlit as st

SYSTEM_PROMPT = """
You are CUAB Buddy, the friendly assistant for Crescent University. Speak warmly and clearly, like a human.
"""

@st.cache_resource()
def setup_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open("data/qa_data.json", "r", encoding="utf-8") as f:

        data = json.load(f)
    questions = [item["question"] for item in data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return model, data, embeddings

model, qa_data, qa_embeddings = setup_resources()
