from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config

FALLBACK_RESPONSE = (
    "I'm having trouble processing your request right now. "
    "Please try again in a moment."
)

FALLBACK_RESPONSE_HI = (
    "मुझे अभी आपका अनुरोध संसाधित करने में कठिनाई हो रही है। "
    "कृपया कुछ क्षण बाद पुनः प्रयास करें।"
)


def get_llm(temperature=None):
    return ChatGoogleGenerativeAI(
        model=config.MODEL_NAME,
        google_api_key=config.GEMINI_API_KEY,
        temperature=temperature or config.LLM_TEMPERATURE,
        convert_system_message_to_human=True,
    )


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )


@retry(
    stop=stop_after_attempt(config.LLM_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _call_llm(llm, messages):
    return llm.invoke(messages)


def generate(messages, temperature=None):
    """Generate a response from the LLM with retry logic."""
    llm = get_llm(temperature=temperature)
    try:
        response = _call_llm(llm, messages)
        return response.content
    except Exception:
        return FALLBACK_RESPONSE


def generate_json(messages, temperature=0.2):
    """Generate a structured JSON response (for intent classification)."""
    llm = get_llm(temperature=temperature)
    try:
        response = _call_llm(llm, messages)
        return response.content
    except Exception:
        return None
