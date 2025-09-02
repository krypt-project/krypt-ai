import re

from flask import Blueprint, jsonify, request
from utils.auth import require_scope
from sentence_transformers import SentenceTransformer, util

tags_bp = Blueprint("tags", __name__)

# Charging modele
model = SentenceTransformer("all-distilroberta-v1")

# Labels
possible_labels = [
    # Général / Organisation
    "Travail", "Personnel", "Urgent", "Important", "À faire", "Idée", "Projet", "Note", "Résumé", "Compte rendu",
    "Réunion", "Appel", "Tâche", "Planification", "Objectif", "Agenda", "Événement", "Rappel", "Checklist", "Priorité",

    # Études / Académique
    "Cours magistral", "Travaux dirigés", "Travaux pratiques", "Mémoire", "Thèse", "Examen", "Recherche",
    "Lecture", "Synthèse", "Rapport", "Présentation", "Projet académique", "Étude de cas", "Équations différentielles",
    "Statistiques", "Analyse", "Expérimentation", "Observation", "Révision", "Apprentissage", "Tutoriel", "Exercice",

    # Sciences
    "Mathématiques", "Physique", "Chimie", "Biologie", "Médecine", "Informatique", "Astronomie", "Géologie",
    "Écologie", "Océanographie", "Psychologie", "Sociologie", "Anthropologie", "Histoire", "Géographie",
    "Philosophie", "Économie", "Politique", "Droit", "Linguistique", "Neurosciences", "Médecine vétérinaire",

    # Technologie / Informatique
    "Programmation", "Machine Learning", "Intelligence Artificielle", "Data Science", "Big Data", "Cybersécurité",
    "Réseaux", "Base de données", "Développement Web", "Développement Mobile", "Logiciel", "Algorithmique",
    "Robotique", "Automatisation", "IoT", "Blockchain", "Cryptographie", "Cloud", "DevOps", "Tests", "UX/UI",

    # Mathématiques avancées
    "Algèbre", "Analyse", "Géométrie", "Probabilités", "Statistiques", "Topologie", "Combinatoire", "Théorie des graphes",

    # Sciences expérimentales
    "Chimie organique", "Chimie inorganique", "Chimie analytique", "Physique théorique", "Physique appliquée",
    "Biologie moléculaire", "Biologie cellulaire", "Génétique", "Microbiologie", "Neurosciences",

    # Créativité / Art / Loisirs
    "Musique", "Chant", "Instrument", "Peinture", "Dessin", "Sculpture", "Photographie", "Vidéo", "Cinéma",
    "Écriture", "Poésie", "Théâtre", "Danse", "Design", "Mode", "Architecture", "Jeux vidéo", "Jeux de société",
    "DIY", "Cuisine", "Voyage", "Sport", "Fitness", "Méditation", "Bien-être", "Yoga", "Lecture loisir", "Podcast", "Blog",

    # Vie pratique
    "Finances", "Budget", "Facture", "Banque", "Assurance", "Santé", "Alimentation", "Courses", "Maison",
    "Jardinage", "Animaux", "Transport", "Voiture", "Immobilier", "Administration", "Legal", "Contrat", "Assurance",

    # Communication / Social
    "Email", "Message", "Discussion", "Networking", "Collaboration", "Team", "Client", "Fournisseur", "Partenaire",
    "Feedback", "Rapport d’activité", "Compte rendu", "Pitch", "Présentation professionnelle",

    # Développement personnel
    "Lecture développement personnel", "Objectif personnel", "Habitude", "Motivation", "Mindset", "Journal", 
    "Réflexion", "Plan d’action", "Compétence", "Apprentissage continu",

    # Arts / Histoire / Littérature
    "Art", "Histoire de l'art", "Mouvement artistique", "Dada", "Surréalisme", "Littérature", "Poésie", 
    "Manifestes", "Avant-garde", "Théorie artistique", "Critique d'art", "Culture", "Expression artistique"

    # Autres / Misc
    "Inclassable", "Divers", "Archive", "Projet futur", "Idée créative", "Inspiration", "Recherche avancée"
]


# Pre-calculate embeddings tags
tags_embeddings = model.encode(possible_labels, convert_to_tensor=True)

# Hyperparameters
THRESHOLD = 0.4
TOP_N = 5

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
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

    tags = {tag: score for tag, score in scored_tags[:TOP_N*2] if score > THRESHOLD}
    if not tags:
        tags = dict(scored_tags[:TOP_N])

    return jsonify(tags)
