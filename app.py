
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random

app = Flask(__name__, static_folder='static')

# Load model and dataset
model = SentenceTransformer('all-MiniLM-L6-v2')

qa_pairs = []
with open("UNIVERSITY DATASET.txt", 'r', encoding='utf-8') as file:
    question, answer = None, None
    for line in file:
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question and answer:
                qa_pairs.append((question, answer))
                question, answer = None, None

dataset = pd.DataFrame(qa_pairs, columns=["question", "response"])
question_embeddings = model.encode(dataset['question'].tolist(), convert_to_tensor=True)

def find_response(user_input, threshold=0.3):
    user_input = user_input.strip().lower()
    greetings = ["hi", "hello", "hey", "hi there", "greetings"]

    if user_input in greetings:
        return random.choice(["Hello!", "Hi!", "Hey there!", "Greetings!"])

    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    top_score = torch.max(cos_scores).item()
    top_index = torch.argmax(cos_scores).item()

    if top_score < threshold:
        return "I'm not sure how to answer that. Can you rephrase?"

    return dataset.iloc[top_index]['response']

# Serve static index.html
@app.route("/")
def serve_ui():
    return send_from_directory("static", "index.html")

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = find_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
