import re

from flask import Blueprint, jsonify, request
from utils.auth import require_scope
from sentence_transformers import SentenceTransformer, util

tags_bp = Blueprint("tags", __name__)

# Charging modele
model = SentenceTransformer("all-distilroberta-v1")

# Labels
possible_labels = [
    "Travail", "Personnel", "Urgent", "Idée", "Projet", "Lecture", "Étude", "Réunion",
    "Examen", "Recherche", "Résumé", "Compte rendu",
    "Mathématiques", "Physique", "Chimie", "Biologie", "Médecine",
    "Programmation", "Machine Learning", "Intelligence Artificielle",
    "Cours magistral", "Travaux dirigés", "Travaux pratiques", "Mémoire", "Équations différentielles",
]

# Pre-calculate embeddings tags
tags_embeddings = model.encode(possible_labels, convert_to_tensor=True)

# Hyperparameters
THRESHOLD = 0.35
TOP_N = 5

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@tags_bp.route("/generate-tags", methods=["POST"])
@require_scope("notes:tags")
def GenerateTags():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_embedding = model.encode(preprocess(text), convert_to_tensor=True)
    scores = util.cos_sim(text_embedding, tags_embeddings)[0].tolist()

    scored_tags = sorted(
        [(tag, float(score)) for tag, score in zip(possible_labels, scores)],
        key=lambda x: x[1],
        reverse=True
    )

    if data.get("debug"):
        return jsonify(dict(scored_tags))

    tags = {tag: score for tag, score in scored_tags[:TOP_N] if score > THRESHOLD}
    if not tags:
        tags = dict(scored_tags[:TOP_N])

    return jsonify(tags)
