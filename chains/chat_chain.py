from langchain_core.messages import HumanMessage, SystemMessage

from chains.prompts import (
    SYSTEM_PROMPT,
    RESPONSE_WITH_CONTEXT_PROMPT,
    RESPONSE_WITHOUT_CONTEXT_PROMPT,
)
from services import intent_classifier, memory, retrieval, language, llm_service


def handle_chat(session_id, message, user_profile):
    """Main orchestration pipeline for handling a chat request.

    Flow:
    1. Get/create session and build astro profile
    2. Translate query if Hindi
    3. Classify intent (decide whether to retrieve)
    4. Conditionally retrieve from ChromaDB
    5. Build prompt with profile, history, context, language
    6. Generate response via LLM
    7. Store turn in memory
    8. Return structured output

    Args:
        session_id: Unique session identifier
        message: User's message
        user_profile: Dict with user's birth details and preferences

    Returns:
        Dict with response, zodiac, context_used, retrieval_used
    """
    # 1. Get or create session with enriched astro profile
    session = memory.get_or_create_session(session_id, user_profile)
    profile = session["user_profile"]
    preferred_lang = profile.get("preferred_language", "en")

    # 2. Translate query to English for retrieval (if Hindi)
    search_query = language.translate_query_for_retrieval(message, preferred_lang)

    # 3. Classify intent — decide whether to retrieve
    context_window = memory.get_context_window(session_id)
    intent = intent_classifier.classify_intent(
        search_query, context_window["recent_history"]
    )

    # 4. Conditionally retrieve context from ChromaDB
    contexts = []
    retrieval_used = False
    if intent.needs_retrieval:
        try:
            contexts = retrieval.retrieve_context(
                query=search_query,
                topic=intent.topic if intent.topic != "general" else None,
                zodiac=profile.get("zodiac"),
            )
            contexts = retrieval.trim_context(contexts)
            retrieval_used = len(contexts) > 0
        except Exception:
            # Graceful degradation — proceed without retrieval
            retrieval_used = False

    # 5. Build the prompt
    prompt_text = _build_prompt(
        profile=profile,
        context_window=context_window,
        contexts=contexts,
        message=message,
        retrieval_used=retrieval_used,
        preferred_lang=preferred_lang,
    )

    # 6. Call LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    response = llm_service.generate(messages)

    # Handle fallback for Hindi
    if preferred_lang == "hi" and response == llm_service.FALLBACK_RESPONSE:
        response = llm_service.FALLBACK_RESPONSE_HI

    # 7. Store turn in memory
    memory.add_turn(session_id, "user", message)
    memory.add_turn(session_id, "assistant", response)

    # 8. Build and return structured output
    context_used = list({ctx["source"] for ctx in contexts}) if contexts else []

    return {
        "response": response,
        "zodiac": profile.get("zodiac", "Unknown"),
        "moon_sign": profile.get("moon_sign", "Unknown"),
        "nakshatra": profile.get("nakshatra", "Unknown"),
        "context_used": context_used,
        "retrieval_used": retrieval_used,
        "intent": {
            "topic": intent.topic,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning,
        },
    }


def _build_prompt(profile, context_window, contexts, message, retrieval_used, preferred_lang):
    """Build the final prompt for LLM generation."""

    # Format summary section
    summary = context_window.get("summary", "")
    summary_section = f"CONVERSATION SUMMARY (older turns):\n{summary}" if summary else ""

    # Format recent history
    recent = context_window.get("recent_history", [])
    if recent:
        history_text = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Astrologer'}: {msg['content']}"
            for msg in recent[-6:]  # Last 3 turns for prompt
        )
    else:
        history_text = "(This is the start of the conversation)"

    # Language instruction
    lang_instruction = language.get_language_instruction(preferred_lang)

    if retrieval_used and contexts:
        # Format retrieved context
        context_text = "\n\n".join(
            f"[{ctx['source']}] (relevance: {ctx['score']:.2f}): {ctx['content']}"
            for ctx in contexts
        )

        return RESPONSE_WITH_CONTEXT_PROMPT.format(
            system_prompt="",  # System prompt is in SystemMessage
            name=profile.get("name", "User"),
            zodiac=profile.get("zodiac", "Unknown"),
            moon_sign=profile.get("moon_sign", "Unknown"),
            nakshatra=profile.get("nakshatra", "Unknown"),
            age=profile.get("age", "Unknown"),
            birth_place=profile.get("birth_place", "Unknown"),
            summary_section=summary_section,
            recent_history=history_text,
            retrieved_context=context_text,
            language_instruction=lang_instruction,
            message=message,
        )
    else:
        return RESPONSE_WITHOUT_CONTEXT_PROMPT.format(
            system_prompt="",
            name=profile.get("name", "User"),
            zodiac=profile.get("zodiac", "Unknown"),
            moon_sign=profile.get("moon_sign", "Unknown"),
            nakshatra=profile.get("nakshatra", "Unknown"),
            age=profile.get("age", "Unknown"),
            birth_place=profile.get("birth_place", "Unknown"),
            summary_section=summary_section,
            recent_history=history_text,
            language_instruction=lang_instruction,
            message=message,
        )
