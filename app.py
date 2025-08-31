from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Modèle rapide
model = SentenceTransformer("all-distilroberta-v1")

# Tags possibles
possible_labels = [
    "Travail", "Personnel", "Urgent", "Idée", "Projet", "Lecture", "Étude", "Réunion",
    "Examen", "Recherche", "Résumé", "Compte rendu",
    "Mathématiques", "Physique", "Chimie", "Biologie", "Médecine",
    "Programmation", "Machine Learning", "Intelligence Artificielle",
    "Cours magistral", "Travaux dirigés", "Travaux pratiques", "Mémoire"
]

# Pré-calcul embeddings des tags
tags_embeddings = model.encode(possible_labels, convert_to_tensor=True)

SEUIL = 0.25
TOP_N = 5

@app.route("/generate-tags", methods=["POST"])
def generate_tags():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_embedding = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(text_embedding, tags_embeddings)[0].tolist()

    scored_tags = sorted(
        [(tag, float(score)) for tag, score in zip(possible_labels, scores)],
        key=lambda x: x[1],
        reverse=True
    )

    # Renvoyer tous les tags > SEUIL, sinon fallback sur le max
    tags = {tag: score for tag, score in scored_tags if score > SEUIL}
    if not tags:
        tags = dict(scored_tags[:TOP_N])

    return jsonify(tags)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
