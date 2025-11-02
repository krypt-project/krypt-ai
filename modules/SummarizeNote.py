from flask import Blueprint, request, jsonify
from transformers import pipeline
from utils.auth import require_scope
import math

summary_bp = Blueprint('summarize', __name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

MAX_TOKENS = 1022  # Limite de BART-large-CNN

def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS):
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i+max_tokens]
        if len(chunk_ids) == 0:
            continue
        # s'assurer que la longueur ne dépasse pas max_tokens
        chunk_ids = chunk_ids[:max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks

@summary_bp.route('/generate-summary', methods=['POST'])
@require_scope("notes:summarize")
def generate_summary():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        chunks = chunk_text(text, tokenizer)
        
        summaries = []
        for chunk in chunks:
            if len(tokenizer(chunk)["input_ids"]) > MAX_TOKENS:
                chunk = tokenizer.decode(tokenizer(chunk)["input_ids"][:MAX_TOKENS], skip_special_tokens=True)
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary.strip())
        
        final_summary = " ".join(summaries)
        return jsonify({"summary": final_summary})

    except Exception as e:
        import traceback
        print("ERROR in /generate-summary:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
