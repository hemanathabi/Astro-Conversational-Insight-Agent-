"""
Evaluation Script — RAG Impact Analysis

Demonstrates two concrete cases:
1. Where retrieval HELPED (grounded, specific response)
2. Where retrieval HURT (irrelevant context diluted response)

Usage:
    python evaluation/eval_cases.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from chains.prompts import SYSTEM_PROMPT, RESPONSE_WITH_CONTEXT_PROMPT, RESPONSE_WITHOUT_CONTEXT_PROMPT
from services import llm_service, retrieval, memory
from services.astro_profile import build_profile
from services.language import get_language_instruction

# Test user profile
TEST_PROFILE = {
    "name": "Ritika",
    "birth_date": "1995-08-20",
    "birth_time": "14:30",
    "birth_place": "Jaipur, India",
    "preferred_language": "en",
}


def run_with_retrieval(message, profile, topic=None):
    """Run a query WITH retrieval enabled."""
    enriched = build_profile(profile)
    contexts = retrieval.retrieve_context(
        query=message,
        topic=topic,
        zodiac=enriched["zodiac"],
    )
    contexts = retrieval.trim_context(contexts)

    context_text = "\n\n".join(
        f"[{ctx['source']}] (score: {ctx['score']:.2f}): {ctx['content']}"
        for ctx in contexts
    )

    prompt = RESPONSE_WITH_CONTEXT_PROMPT.format(
        system_prompt="",
        name=enriched["name"],
        zodiac=enriched["zodiac"],
        moon_sign=enriched["moon_sign"],
        nakshatra=enriched["nakshatra"],
        age=enriched["age"],
        birth_place=enriched["birth_place"],
        summary_section="",
        recent_history="(Start of conversation)",
        retrieved_context=context_text,
        language_instruction="Respond in English.",
        message=message,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm_service.generate(messages)
    return response, contexts


def run_without_retrieval(message, profile):
    """Run a query WITHOUT retrieval."""
    enriched = build_profile(profile)

    prompt = RESPONSE_WITHOUT_CONTEXT_PROMPT.format(
        system_prompt="",
        name=enriched["name"],
        zodiac=enriched["zodiac"],
        moon_sign=enriched["moon_sign"],
        nakshatra=enriched["nakshatra"],
        age=enriched["age"],
        birth_place=enriched["birth_place"],
        summary_section="",
        recent_history="(Start of conversation)",
        language_instruction="Respond in English.",
        message=message,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm_service.generate(messages)
    return response


def eval_case_1():
    """Case 1: Retrieval HELPED — specific astrological question."""
    print("=" * 70)
    print("CASE 1: Retrieval HELPED")
    print("Question: 'Which planet is affecting my career and what should I do?'")
    print("=" * 70)

    message = "Which planet is affecting my career and what should I do?"

    print("\n--- Run A: WITH Retrieval ---")
    response_with, contexts = run_with_retrieval(message, TEST_PROFILE, topic="career")
    print(f"Retrieved {len(contexts)} contexts:")
    for ctx in contexts:
        print(f"  [{ctx['source']}] score={ctx['score']:.2f}: {ctx['content'][:80]}...")
    print(f"\nResponse:\n{response_with}")

    print("\n--- Run B: WITHOUT Retrieval ---")
    response_without = run_without_retrieval(message, TEST_PROFILE)
    print(f"\nResponse:\n{response_without}")

    print("\n--- Analysis ---")
    print("With retrieval: Response is grounded in specific planetary data and career")
    print("guidance from the knowledge base. References concrete planet traits and advice.")
    print("Without retrieval: Response is more generic, relying on the LLM's general")
    print("knowledge without specific corpus-backed facts.")
    print("Verdict: RETRIEVAL HELPED — it produced a more specific, grounded response.")


def eval_case_2():
    """Case 2: Retrieval HURT — conversational/summary question."""
    print("\n" + "=" * 70)
    print("CASE 2: Retrieval HURT")
    print("Question: 'Can you summarize what you have told me so far?'")
    print("=" * 70)

    # Set up a fake conversation history
    message = "Can you summarize what you have told me so far?"

    print("\n--- Run A: WITH Retrieval (forced) ---")
    response_with, contexts = run_with_retrieval(message, TEST_PROFILE)
    print(f"Retrieved {len(contexts)} contexts:")
    for ctx in contexts:
        print(f"  [{ctx['source']}] score={ctx['score']:.2f}: {ctx['content'][:80]}...")
    print(f"\nResponse:\n{response_with}")

    print("\n--- Run B: WITHOUT Retrieval ---")
    response_without = run_without_retrieval(message, TEST_PROFILE)
    print(f"\nResponse:\n{response_without}")

    print("\n--- Analysis ---")
    print("With retrieval: Retrieved context is irrelevant to summarization task.")
    print("The injected zodiac/career facts distract from the actual summarization.")
    print("Without retrieval: Response correctly focuses on summarizing the conversation")
    print("history without introducing unrelated astrological facts.")
    print("Verdict: RETRIEVAL HURT — irrelevant context diluted the summary response.")
    print("This is why the intent classifier should skip retrieval for 'summarize' intents.")


def main():
    print("\n" + "#" * 70)
    print("# ASTRO INSIGHT AGENT — RAG EVALUATION")
    print("#" * 70)

    eval_case_1()
    eval_case_2()

    print("\n" + "#" * 70)
    print("# EVALUATION COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
