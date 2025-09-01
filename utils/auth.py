import os
import jwt
import dotenv

from flask import request, jsonify
from jwt import InvalidTokenError

dotenv.load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

def require_scope(required_scope):
    def decorator(func):
        def wrapper(*args, **kwargs):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid Authorization header"}), 401

            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                scopes = payload.get("scope", [])
                if isinstance(scopes, str):
                    scopes = [scopes]
                if required_scope not in scopes:
                    return jsonify({"error": "Forbidden, missing scope"}), 403
            except InvalidTokenError as e:
                return jsonify({"error": f"Invalid token: {str(e)}"}), 401

            return func(*args, **kwargs)
        return wrapper
    return decorator
