"""Tests for intent classification logic."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.intent_classifier import classify_intent


class TestRuleBasedClassification:
    """Test the rule-based fast path of intent classification."""

    def test_career_keyword(self):
        result = classify_intent("How will my career be this month?")
        assert result.needs_retrieval is True
        assert result.topic == "career"

    def test_love_keyword(self):
        result = classify_intent("What about my love life?")
        assert result.needs_retrieval is True
        assert result.topic == "love"

    def test_planet_keyword(self):
        result = classify_intent("Which planet affects me most?")
        assert result.needs_retrieval is True
        assert result.topic == "planetary"

    def test_spiritual_keyword(self):
        result = classify_intent("Give me spiritual guidance")
        assert result.needs_retrieval is True
        assert result.topic == "spiritual"

    def test_nakshatra_keyword(self):
        result = classify_intent("Tell me about my nakshatra")
        assert result.needs_retrieval is True
        assert result.topic == "nakshatra"

    def test_zodiac_keyword(self):
        result = classify_intent("What are Leo traits?")
        assert result.needs_retrieval is True
        assert result.topic == "personality"

    def test_summarize_no_retrieval(self):
        result = classify_intent("Can you summarize what you told me?")
        assert result.needs_retrieval is False
        assert result.topic == "conversational"

    def test_greeting_no_retrieval(self):
        result = classify_intent("Hello!")
        assert result.needs_retrieval is False
        assert result.topic == "conversational"

    def test_thanks_no_retrieval(self):
        result = classify_intent("Thank you so much!")
        assert result.needs_retrieval is False
        assert result.topic == "conversational"

    def test_repeat_no_retrieval(self):
        result = classify_intent("Can you repeat that again?")
        assert result.needs_retrieval is False
        assert result.topic == "conversational"

    def test_bye_no_retrieval(self):
        result = classify_intent("Okay bye!")
        assert result.needs_retrieval is False
        assert result.topic == "conversational"

    def test_what_did_you_say_no_retrieval(self):
        result = classify_intent("What did you tell me about career?")
        assert result.needs_retrieval is False
        assert result.topic == "conversational"


class TestIntentConfidence:
    """Test confidence levels of intent classification."""

    def test_keyword_match_confidence(self):
        result = classify_intent("Tell me about Mars planet effects")
        assert result.confidence >= 0.8

    def test_no_retrieve_confidence(self):
        result = classify_intent("Summarize everything")
        assert result.confidence >= 0.8


class TestAstroProfileIntegration:
    """Test astro profile calculations."""

    def test_sun_sign_leo(self):
        from services.astro_profile import get_sun_sign
        assert get_sun_sign("1995-08-20") == "Leo"

    def test_sun_sign_aries(self):
        from services.astro_profile import get_sun_sign
        assert get_sun_sign("1990-04-05") == "Aries"

    def test_sun_sign_capricorn_december(self):
        from services.astro_profile import get_sun_sign
        assert get_sun_sign("1990-12-25") == "Capricorn"

    def test_sun_sign_capricorn_january(self):
        from services.astro_profile import get_sun_sign
        assert get_sun_sign("1990-01-10") == "Capricorn"

    def test_age_calculation(self):
        from services.astro_profile import get_age
        age = get_age("1995-08-20")
        assert age is not None
        assert age >= 30  # As of 2026

    def test_build_profile(self):
        from services.astro_profile import build_profile
        profile = build_profile({
            "name": "Test",
            "birth_date": "1995-08-20",
            "birth_time": "14:30",
            "birth_place": "Delhi",
        })
        assert profile["zodiac"] == "Leo"
        assert profile["moon_sign"] is not None
        assert profile["nakshatra"] is not None
        assert profile["age"] is not None
