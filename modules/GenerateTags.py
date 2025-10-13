import re
import dotenv
import os
import psycopg2

from psycopg2.extras import RealDictCursor
from flask import Blueprint, jsonify, request
from utils.auth import require_scope
from sentence_transformers import SentenceTransformer, util


dotenv.load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Database connexion and tags request
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def get_all_tags():
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT name FROM tag_table;")
            rows = cur.fetchall()
    return [row["name"] for row in rows]

tags_bp = Blueprint("tags", __name__)

# Charging modele
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

# Hyperparameters
THRESHOLD = 0.4
TOP_N = 5

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# Generate Tags route
@tags_bp.route("/generate-tags", methods=["POST"])
@require_scope("notes:tags")
def GenerateTags():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Charge tags from DB
    possible_labels = get_all_tags()
    print(possible_labels)

    if not possible_labels:
        return jsonify({"error": "No tags available in database"}), 500

    # Dynamiques embeddings
    tags_embeddings = model.encode(possible_labels, convert_to_tensor=True)
    text_embedding = model.encode(preprocess(text), convert_to_tensor=True)
    scores = util.cos_sim(text_embedding, tags_embeddings)[0].tolist()

    scored_tags = sorted(
        [(str(tag), float(score)) for tag, score in zip(possible_labels, scores)],
        key=lambda x: x[1],
        reverse=True
    )

    if data.get("debug"):
        return jsonify([
            {"tag": tag, "score": score} for tag, score in scored_tags
        ])

    tags = {tag: score for tag, score in scored_tags[:TOP_N*2] if score > THRESHOLD}
    if not tags:
        tags = dict(scored_tags[:TOP_N])

    tags = [
        {"tag": str(tag), "score": float(score)}
        for tag, score in scored_tags[:TOP_N]
    ]

    return jsonify(tags)

# Add Tags route
@tags_bp.route("/add-tag", methods=["POST"])
@require_scope("notes:tags")
def add_tag():
    data = request.get_json()
    tag = data.get("tag", "").strip()

    if not tag:
        return jsonify({"error": "Tag name required"}), 400

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO tag_table (name) VALUES (%s) ON CONFLICT DO NOTHING;", (tag,))
                conn.commit()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "tag": tag})