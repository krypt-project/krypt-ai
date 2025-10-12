from flask import Blueprint, request, jsonify
from transformers import pipeline
from utils.auth import require_scope

chatbot_bp = Blueprint("chatbot", __name__)

# Chargement du modèle globalement (évite de recharger à chaque requête)
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)

@chatbot_bp.route("/chatbot", methods=["POST"])
@require_scope("chatbot:generate")
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    try:
        # Préparer le prompt pour un comportement type "assistant"
        prompt = f"Utilisateur: {user_message}\nAssistant:"

        response = generator(
            prompt,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            num_return_sequences=1,
            truncation=True
        )[0]["generated_text"]

        # Nettoyage : on garde seulement le texte après "Assistant:"
        if "Assistant:" in response:
            clean_resp = response.split("Assistant:", 1)[-1].strip()
        else:
            clean_resp = response[len(user_message):].strip()

        return jsonify({"reply": clean_resp})

    except Exception as e:
        import traceback
        print("ERROR in /chatbot : ", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
