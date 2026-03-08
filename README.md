# Astro Conversational Insight Agent

A multi-turn conversational AI service for personalized astrology guidance, built with **Flask**, **LangChain**, **ChromaDB**, and **Google Gemini**.

## Features

- **Multi-turn conversation** with session-based memory (sliding window + summarization)
- **Intent-aware RAG** — retrieves astrological knowledge only when it adds value
- **Personalization** — zodiac sign, moon sign, nakshatra, and age from birth details
- **Hindi language support** — responds in Hindi (Devanagari) when requested
- **9 planets** including Rahu & Ketu with malefic/benefic classification
- **27 nakshatras** with full mapping
- **Similarity scoring & context trimming** for cost-efficient retrieval
- **Comprehensive evaluation** with retrieval-helped vs retrieval-hurt cases

## Architecture

```
POST /chat
  |
  v
Input Validation (Flask)
  |
  v
Session Management (get/create session, build astro profile)
  |
  v
Query Translation (Hindi -> English if needed)
  |
  v
Intent Classification (rule-based -> LLM fallback)
  |
  v
[If needs_retrieval] ChromaDB Vector Search -> Scoring -> Trimming
  |
  v
Prompt Construction (profile + history + summary + context + language)
  |
  v
LLM Generation (Gemini via LangChain, with retries)
  |
  v
Memory Update (add turn, summarize if window exceeded)
  |
  v
Structured Response { response, zodiac, context_used, retrieval_used }
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | Flask |
| LLM Orchestration | LangChain |
| LLM | Google Gemini 2.5 Pro |
| Vector Database | ChromaDB |
| Embeddings | Google Generative AI Embeddings |
| Language | Python 3.10+ |

## Project Structure

```
├── app.py                      # Flask API (/chat, /health)
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── .env                        # API key (not committed)
│
├── data/                       # Knowledge base (6 files)
│   ├── zodiac_traits.json      # 12 zodiac signs
│   ├── planetary_impacts.json  # 9 planets (incl. Rahu/Ketu)
│   ├── career_guidance.txt     # Career advice
│   ├── love_guidance.txt       # Relationship advice
│   ├── spiritual_guidance.txt  # Spiritual guidance
│   └── nakshatra_mapping.json  # 27 nakshatras
│
├── services/                   # Business logic
│   ├── astro_profile.py        # Zodiac/moon sign/nakshatra calculation
│   ├── intent_classifier.py    # Intent-aware retrieval decisions
│   ├── retrieval.py            # ChromaDB search + scoring
│   ├── memory.py               # Session store + sliding window
│   ├── llm_service.py          # Gemini abstraction + retries
│   └── language.py             # Hindi/English toggle
│
├── chains/                     # LangChain orchestration
│   ├── chat_chain.py           # Main pipeline
│   └── prompts.py              # Prompt templates
│
├── knowledge/
│   └── ingest.py               # ChromaDB ingestion script
│
├── evaluation/                 # RAG evaluation
│   ├── eval_cases.py           # Automated eval script
│   └── eval_results.md         # Written analysis
│
└── tests/                      # Tests
    ├── test_api.py             # API endpoint tests
    └── test_intent.py          # Intent + profile tests
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Ingest Knowledge Base

```bash
python knowledge/ingest.py
```

This embeds all data files into ChromaDB (takes ~1 minute).

### 4. Run the Server

```bash
python app.py
```

Server starts at `http://localhost:5000`.

## API Usage

### POST /chat

**Request:**
```json
{
  "session_id": "abc-123",
  "message": "How will my month be in career?",
  "user_profile": {
    "name": "Ritika",
    "birth_date": "1995-08-20",
    "birth_time": "14:30",
    "birth_place": "Jaipur, India",
    "preferred_language": "hi"
  }
}
```

**Response:**
```json
{
  "response": "आपके लिए यह महीना करियर में अवसर लेकर आ रहा है...",
  "zodiac": "Leo",
  "moon_sign": "Aquarius",
  "nakshatra": "Dhanishtha",
  "context_used": ["career_guidance", "planetary_impacts"],
  "retrieval_used": true,
  "intent": {
    "topic": "career",
    "confidence": 0.85,
    "reasoning": "Matched retrieve keywords for topic: career"
  }
}
```

### GET /health

Returns service health status.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run evaluation
python evaluation/eval_cases.py
```

## Design Decisions

### Intent-Aware RAG (Not Always-On)
The system uses a **two-stage intent classifier**:
1. **Rule-based fast path** — keyword matching handles ~70% of queries at zero cost
2. **LLM fallback** — Gemini classifies ambiguous queries

This prevents retrieval on conversational queries ("summarize", "thanks") where injecting knowledge base context would hurt response quality.

### Memory Management
- **Sliding window**: Keeps last 10 conversation turns
- **Automatic summarization**: When window overflows, older turns are summarized via LLM
- **Bounded growth**: Summary is capped at ~500 tokens

### Retrieval Scoring
- **Similarity threshold** (0.35): Drops low-relevance results
- **Context trimming**: Caps retrieved context at 2000 tokens
- **Metadata filtering**: Filters by topic and zodiac sign for precision

### Hindi Support
- English knowledge base with **query translation** at boundaries
- Hindi queries are translated to English for vector search
- Response language instruction forces Gemini to reply in Devanagari

### Moon Sign & Nakshatra
- Approximate calculation from birth date (real Vedic calculation requires Swiss Ephemeris)
- Architecture supports plugging in a real ephemeris library
