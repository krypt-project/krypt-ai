from flask import Blueprint, request, jsonify
from transformers import pipeline
from utils.auth import require_scope

summary_bp = Blueprint('summarize', __name__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@summary_bp.route('/generate-summary', methods=['POST'])
@require_scope("notes:summarize")
def generate_summary():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        summary = summarizer(
            text,
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]['summary_text']

        return jsonify({"summary": summary.strip()})
    except Exception as e:
        import traceback
        print("🔥 ERROR in /generate-summary:", traceback.format_exc())  # log complet dans la console
        return jsonify({"error": str(e)}), 500