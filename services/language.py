from langchain_core.messages import HumanMessage, SystemMessage
from services import llm_service


def get_language_instruction(preferred_language):
    """Get prompt instruction for response language.

    Args:
        preferred_language: "hi" for Hindi, "en" for English (default)

    Returns:
        String instruction to append to the prompt
    """
    if preferred_language == "hi":
        return (
            "IMPORTANT: You MUST respond entirely in Hindi using Devanagari script (हिन्दी). "
            "Do not use English except for proper nouns like zodiac sign names (Leo, Mars, etc). "
            "Make your response natural and conversational in Hindi."
        )
    return "Respond in English."


def translate_query_for_retrieval(query, preferred_language):
    """Translate Hindi query to English for vector search.

    Since the knowledge base is in English, we translate Hindi queries
    to English before performing retrieval.

    Args:
        query: User's message (possibly in Hindi)
        preferred_language: "hi" or "en"

    Returns:
        English version of the query for retrieval
    """
    if preferred_language != "hi":
        return query

    # Check if the query is already mostly in English (ASCII check)
    ascii_ratio = sum(1 for c in query if ord(c) < 128) / max(len(query), 1)
    if ascii_ratio > 0.8:
        return query

    # Translate Hindi to English for retrieval purposes
    messages = [
        SystemMessage(content="You are a translator. Translate the given Hindi text to English. Output ONLY the translation, nothing else."),
        HumanMessage(content=query),
    ]

    translated = llm_service.generate(messages, temperature=0.1)
    return translated if translated else query
