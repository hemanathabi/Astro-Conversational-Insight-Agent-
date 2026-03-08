from services import llm_service
from services.astro_profile import build_profile
from langchain_core.messages import HumanMessage, SystemMessage
import config

# In-memory session store
_sessions = {}


def get_or_create_session(session_id, user_profile):
    """Get existing session or create a new one with enriched astro profile.

    Args:
        session_id: Unique session identifier
        user_profile: Dict with name, birth_date, birth_time, birth_place, preferred_language

    Returns:
        Session dict with user_profile, conversation_history, summary, turn_count
    """
    if session_id not in _sessions:
        enriched_profile = build_profile(user_profile)
        _sessions[session_id] = {
            "user_profile": enriched_profile,
            "conversation_history": [],
            "summary": "",
            "turn_count": 0,
        }
    else:
        # Update profile if provided (user may send updated info)
        existing = _sessions[session_id]
        if user_profile:
            enriched = build_profile(user_profile)
            existing["user_profile"].update(enriched)

    return _sessions[session_id]


def add_turn(session_id, role, content):
    """Add a message to conversation history and manage window size.

    Args:
        session_id: Session identifier
        role: "user" or "assistant"
        content: Message content

    Triggers summarization if history exceeds MAX_CONVERSATION_WINDOW.
    """
    session = _sessions.get(session_id)
    if not session:
        return

    session["conversation_history"].append({
        "role": role,
        "content": content,
    })

    # Increment turn count (a turn = one user + one assistant message)
    if role == "assistant":
        session["turn_count"] += 1

    # Check if we need to summarize older messages
    max_messages = config.MAX_CONVERSATION_WINDOW * 2  # Each turn = 2 messages
    history = session["conversation_history"]

    if len(history) > max_messages:
        _summarize_and_trim(session_id)


def _summarize_and_trim(session_id):
    """Summarize older conversation turns and trim history to window size."""
    session = _sessions[session_id]
    history = session["conversation_history"]
    max_messages = config.MAX_CONVERSATION_WINDOW * 2

    # Messages to summarize (everything beyond the window)
    overflow_count = len(history) - max_messages
    to_summarize = history[:overflow_count]

    # Build text from messages to summarize
    conversation_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in to_summarize
    )

    # Generate summary
    existing_summary = session["summary"]
    summary_prompt = (
        "Summarize the following astrology conversation in 2-3 concise sentences. "
        "Preserve key facts: advice given, topics discussed, and important astrological insights.\n\n"
    )
    if existing_summary:
        summary_prompt += f"Previous summary: {existing_summary}\n\n"
    summary_prompt += f"New conversation to summarize:\n{conversation_text}"

    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes conversations concisely."),
        HumanMessage(content=summary_prompt),
    ]

    new_summary = llm_service.generate(messages, temperature=0.3)

    # Truncate summary if too long (approximate token check)
    if len(new_summary) > config.MAX_SUMMARY_TOKENS * 4:
        new_summary = new_summary[: config.MAX_SUMMARY_TOKENS * 4]

    session["summary"] = new_summary
    session["conversation_history"] = history[overflow_count:]


def get_context_window(session_id):
    """Get the current context window for prompt construction.

    Returns:
        Dict with summary (str) and recent_history (list of messages)
    """
    session = _sessions.get(session_id)
    if not session:
        return {"summary": "", "recent_history": []}

    return {
        "summary": session.get("summary", ""),
        "recent_history": session.get("conversation_history", []),
    }


def get_session(session_id):
    """Get full session data."""
    return _sessions.get(session_id)


def clear_session(session_id):
    """Remove a session from the store."""
    _sessions.pop(session_id, None)
