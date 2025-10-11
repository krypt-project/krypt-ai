from flask import Blueprint, request, jsonify
from transformers import pipeline
from utils.auth import require_scope

chatbot_bp = Blueprint("chatbot", __name__)
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

@chatbot_bp.route("/chatbot", methods=["POST"])
@require_scope("chatbot:generate")
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"error": "Message vide"}), 400
    
    try:
        response = generator(
            user_message,
            max_length=200,
            truncation=True,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]

        clean_resp = response.replace(user_message, "").strip()

        return jsonify({"reply": clean_resp})
    
    except Exception as e:
        import traceback
        print("ERROR in /chatbot : ", traceback.format_exc())
        return jsonify({"error": str(e)}), 500