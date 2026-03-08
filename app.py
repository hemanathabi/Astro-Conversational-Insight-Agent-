"""
Astro Conversational Insight Agent — Flask API with Swagger UI

Endpoints:
    POST /chat    — Multi-turn astrology chat
    GET  /health  — Health check

Swagger UI available at: http://localhost:5000/
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from datetime import datetime
from chains.chat_chain import handle_chat
import config

app = Flask(__name__)

api = Api(
    app,
    version="1.0",
    title="Astro Conversational Insight Agent",
    description=(
        "A multi-turn conversational AI service for personalized astrology guidance. "
        "Uses RAG (Retrieval-Augmented Generation) with intent-aware retrieval, "
        "session-based memory, and Hindi language support."
    ),
)

# Namespaces
chat_ns = api.namespace("", description="Chat & Health endpoints")

# --- Request/Response Models for Swagger ---

user_profile_model = api.model("UserProfile", {
    "name": fields.String(required=True, description="User's name", example="Ritika"),
    "birth_date": fields.String(required=True, description="Birth date (YYYY-MM-DD)", example="1995-08-20"),
    "birth_time": fields.String(required=False, description="Birth time (HH:MM)", example="14:30"),
    "birth_place": fields.String(required=False, description="Birth place", example="Jaipur, India"),
    "preferred_language": fields.String(
        required=False,
        description="Response language: 'en' (English) or 'hi' (Hindi)",
        example="en",
        enum=["en", "hi"],
    ),
})

chat_request_model = api.model("ChatRequest", {
    "session_id": fields.String(required=True, description="Unique session ID for multi-turn conversation", example="abc-123"),
    "message": fields.String(required=True, description="User's astrology question", example="How will my month be in career?"),
    "user_profile": fields.Nested(user_profile_model, required=True, description="User's birth details"),
})

intent_model = api.model("Intent", {
    "topic": fields.String(description="Classified topic", example="career"),
    "confidence": fields.Float(description="Classification confidence", example=0.85),
    "reasoning": fields.String(description="Why this classification was made", example="Matched retrieve keywords for topic: career"),
})

chat_response_model = api.model("ChatResponse", {
    "response": fields.String(description="Astrologer's response", example="As a Leo ruled by the Sun..."),
    "zodiac": fields.String(description="Sun sign", example="Leo"),
    "moon_sign": fields.String(description="Moon sign (approximate)", example="Aquarius"),
    "nakshatra": fields.String(description="Nakshatra (approximate)", example="Dhanishtha"),
    "context_used": fields.List(fields.String, description="Knowledge sources used", example=["career_guidance", "planetary_impacts"]),
    "retrieval_used": fields.Boolean(description="Whether RAG retrieval was used", example=True),
    "intent": fields.Nested(intent_model, description="Intent classification details"),
})

error_model = api.model("Error", {
    "error": fields.String(description="Error message"),
})

health_model = api.model("Health", {
    "status": fields.String(description="Service status", example="healthy"),
    "model": fields.String(description="LLM model in use", example="gemini-2.5-pro"),
    "service": fields.String(description="Service name", example="Astro Conversational Insight Agent"),
})


def validate_request(data):
    """Validate incoming chat request payload."""
    if not data:
        return False, "Request body is required (JSON)"

    if "session_id" not in data:
        return False, "session_id is required"

    if "message" not in data or not data["message"].strip():
        return False, "message is required and cannot be empty"

    profile = data.get("user_profile")
    if not profile:
        return False, "user_profile is required"

    if "name" not in profile:
        return False, "user_profile.name is required"

    if "birth_date" not in profile:
        return False, "user_profile.birth_date is required"

    try:
        datetime.strptime(profile["birth_date"], "%Y-%m-%d")
    except ValueError:
        return False, "user_profile.birth_date must be in YYYY-MM-DD format"

    if "birth_time" in profile and profile["birth_time"]:
        try:
            datetime.strptime(profile["birth_time"], "%H:%M")
        except ValueError:
            return False, "user_profile.birth_time must be in HH:MM format"

    return True, None


@chat_ns.route("/chat")
class ChatResource(Resource):
    @chat_ns.expect(chat_request_model, validate=False)
    @chat_ns.marshal_with(chat_response_model, code=200, description="Successful response")
    @chat_ns.response(400, "Validation error", error_model)
    @chat_ns.response(500, "Internal server error", error_model)
    def post(self):
        """Send a message to the astrology chatbot.

        Supports multi-turn conversation with session-based memory.
        The system automatically:
        - Determines your zodiac sign, moon sign, and nakshatra from birth details
        - Decides whether to retrieve astrological knowledge (intent-aware RAG)
        - Responds in Hindi if preferred_language is set to "hi"
        - Maintains conversation context across turns using the same session_id
        """
        data = request.get_json(silent=True)

        is_valid, error = validate_request(data)
        if not is_valid:
            api.abort(400, error)

        session_id = data["session_id"]
        message = data["message"].strip()
        user_profile = data["user_profile"]

        if "preferred_language" not in user_profile:
            user_profile["preferred_language"] = "en"

        try:
            result = handle_chat(session_id, message, user_profile)
            return result, 200
        except Exception as e:
            api.abort(500, f"An internal error occurred: {str(e)}")


@chat_ns.route("/health")
class HealthResource(Resource):
    @chat_ns.marshal_with(health_model, code=200, description="Service is healthy")
    def get(self):
        """Check service health status."""
        return {
            "status": "healthy",
            "model": config.MODEL_NAME,
            "service": "Astro Conversational Insight Agent",
        }, 200


if __name__ == "__main__":
    print("=" * 60)
    print("  Astro Conversational Insight Agent")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  ChromaDB: {config.CHROMA_PERSIST_DIR}")
    print(f"  Swagger UI: http://localhost:5000/")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
