from flask import Blueprint, request, jsonify
from google import genai
from google.genai import types
from utils.auth import require_scope

chatbot_bp = Blueprint("chatbot", __name__)
client = genai.Client(api_key="AIzaSyBFfccQO2zSwr_RGUD8cBryMKd-Yp8bkY0")

@chatbot_bp.route("/chatbot", methods=["POST"])
@require_scope("chatbot:generate")
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    try:
        # Génération du texte avec Gemini Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_message],
            config=types.GenerateContentConfig(
                temperature=0.7,
                # thinking_config=types.ThinkingConfig(thinking_budget=0)  # désactive réflexion
            )
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        import traceback
        print("ERROR in /chatbot : ", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
