from flask import Flask
from modules.GenerateTags import tags_bp

app = Flask(__name__)
app.register_blueprint(tags_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)