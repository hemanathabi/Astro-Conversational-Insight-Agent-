import re
import json
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from services import llm_service


@dataclass
class IntentResult:
    needs_retrieval: bool
    topic: str
    confidence: float
    reasoning: str = ""


# Keywords that indicate retrieval is needed, mapped to topics
RETRIEVE_KEYWORDS = {
    # Career
    "career": "career", "job": "career", "work": "career", "profession": "career",
    "business": "career", "promotion": "career", "salary": "career", "office": "career",
    "interview": "career", "employment": "career",
    # Love
    "love": "love", "relationship": "love", "partner": "love", "marriage": "love",
    "romance": "love", "dating": "love", "husband": "love", "wife": "love",
    "boyfriend": "love", "girlfriend": "love", "soulmate": "love",
    # Spiritual
    "spiritual": "spiritual", "meditation": "spiritual", "soul": "spiritual",
    "karma": "spiritual", "mantra": "spiritual", "prayer": "spiritual",
    "chakra": "spiritual", "yoga": "spiritual", "healing": "spiritual",
    # Planetary
    "planet": "planetary", "mars": "planetary", "venus": "planetary",
    "saturn": "planetary", "jupiter": "planetary", "mercury": "planetary",
    "sun": "planetary", "moon": "planetary", "rahu": "planetary", "ketu": "planetary",
    "shani": "planetary", "mangal": "planetary", "surya": "planetary",
    # Personality/Zodiac
    "zodiac": "personality", "sign": "personality", "traits": "personality",
    "personality": "personality", "horoscope": "personality", "compatibility": "personality",
    "aries": "personality", "taurus": "personality", "gemini": "personality",
    "cancer": "personality", "leo": "personality", "virgo": "personality",
    "libra": "personality", "scorpio": "personality", "sagittarius": "personality",
    "capricorn": "personality", "aquarius": "personality", "pisces": "personality",
    # Nakshatra
    "nakshatra": "nakshatra", "star": "nakshatra", "constellation": "nakshatra",
    "ashwini": "nakshatra", "bharani": "nakshatra",
}

# Patterns that indicate NO retrieval needed
NO_RETRIEVE_PATTERNS = [
    r"\b(summarize|summary|recap|overview)\b",
    r"\bwhat did you (say|tell|mention)\b",
    r"\b(repeat|again|rephrase)\b",
    r"\b(thank|thanks|ok|okay|bye|goodbye|hello|hi|hey|namaste)\b",
    r"\bhow are you\b",
    r"\btell me more\b",
    r"\b(who are you|what can you do)\b",
    r"\b(yes|no|sure|alright)\b",
    r"\bwhy (are you|did you) (say|mention|tell)\b",
]

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for an astrology chatbot.
Given the user's message and recent conversation context, determine:
1. Does this question require looking up astrological knowledge from a knowledge base? (yes/no)
2. What topic does it relate to? (career/love/spiritual/planetary/personality/nakshatra/general)
3. Brief reasoning for your decision.

Rules:
- Questions about zodiac traits, planets, career/love/spiritual guidance, or nakshatras NEED retrieval.
- Questions that refer to previous conversation (summarize, repeat, "what did you say") do NOT need retrieval.
- Greetings, acknowledgements, and meta-questions about the bot do NOT need retrieval.
- If the question is vague or could benefit from astrological context, default to retrieval.

User message: "{message}"
Recent conversation (last 2 turns): {recent_context}

Respond ONLY with valid JSON (no markdown, no code blocks):
{{"needs_retrieval": true/false, "topic": "career/love/spiritual/planetary/personality/nakshatra/general", "reasoning": "brief explanation"}}"""


def classify_intent(message, conversation_history=None):
    """Classify user intent to decide whether retrieval is needed.

    Uses a two-stage approach:
    1. Rule-based fast path (keyword/pattern matching)
    2. LLM fallback for ambiguous cases

    Args:
        message: The user's message
        conversation_history: Recent conversation turns (list of dicts)

    Returns:
        IntentResult with needs_retrieval, topic, confidence, reasoning
    """
    message_lower = message.lower().strip()

    # Stage 1: Check NO_RETRIEVE patterns first
    for pattern in NO_RETRIEVE_PATTERNS:
        if re.search(pattern, message_lower):
            return IntentResult(
                needs_retrieval=False,
                topic="conversational",
                confidence=0.9,
                reasoning=f"Matched no-retrieve pattern: {pattern}",
            )

    # Stage 1: Check RETRIEVE keywords
    matched_topics = []
    for keyword, topic in RETRIEVE_KEYWORDS.items():
        if keyword in message_lower:
            matched_topics.append(topic)

    if matched_topics:
        # Use the most common topic among matches
        primary_topic = max(set(matched_topics), key=matched_topics.count)
        return IntentResult(
            needs_retrieval=True,
            topic=primary_topic,
            confidence=0.85,
            reasoning=f"Matched retrieve keywords for topic: {primary_topic}",
        )

    # Stage 2: LLM fallback for ambiguous messages
    return _llm_classify(message, conversation_history)


def _llm_classify(message, conversation_history=None):
    """Use LLM to classify ambiguous intents."""
    recent_context = "None"
    if conversation_history:
        last_turns = conversation_history[-4:]  # Last 2 turns (4 messages)
        recent_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content'][:100]}"
            for msg in last_turns
        )

    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        message=message,
        recent_context=recent_context,
    )

    messages = [
        SystemMessage(content="You are a precise intent classifier. Respond only with JSON."),
        HumanMessage(content=prompt),
    ]

    response = llm_service.generate_json(messages, temperature=0.1)

    if response:
        try:
            # Clean response — remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()

            result = json.loads(cleaned)
            return IntentResult(
                needs_retrieval=result.get("needs_retrieval", True),
                topic=result.get("topic", "general"),
                confidence=0.75,
                reasoning=result.get("reasoning", "LLM classification"),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Default: retrieve to be safe
    return IntentResult(
        needs_retrieval=True,
        topic="general",
        confidence=0.5,
        reasoning="Defaulting to retrieval (LLM classification failed)",
    )
