from flask import Flask
from flask_cors import CORS
from modules.GenerateTags import tags_bp
from modules.SummarizeNote import summary_bp
from modules.Chatbot import chatbot_bp

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

app.register_blueprint(tags_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(chatbot_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)