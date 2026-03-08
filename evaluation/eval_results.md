# RAG Evaluation Results

## Overview

This document presents the evaluation of our intent-aware RAG system through two concrete cases: one where retrieval adds value and one where it detracts from response quality.

## Case 1: Retrieval HELPED

**Query:** "Which planet is affecting my career and what should I do?"
**User Profile:** Leo sun sign, Aquarius moon sign, Dhanishtha nakshatra

### With Retrieval (RAG ON)
- Retrieved contexts from `planetary_impacts` (Sun/Mars/Saturn career influence) and `career_guidance` (actionable advice)
- Response was **specific and grounded**: referenced Sun's rulership of Leo, Mars's drive for career ambition, and Saturn's lessons in discipline
- Included **concrete remedies** from the knowledge base (e.g., "chant Hanuman Chalisa on Tuesdays")
- Provided **actionable career advice** directly from the corpus

### Without Retrieval (RAG OFF)
- Response was **generic**: mentioned general planetary influences without specific data
- Lacked the **corpus-backed specificity** of remedies and zodiac-career connections
- Advice was vague ("focus on your strengths", "be patient")

### Verdict
**Retrieval clearly helped.** The RAG-augmented response was more personalized, specific, and actionable because it was grounded in curated astrological knowledge rather than relying on the LLM's general training data.

---

## Case 2: Retrieval HURT

**Query:** "Can you summarize what you have told me so far?"
**User Profile:** Same as above

### With Retrieval (RAG ON, forced)
- Retrieved contexts were **semantically matched but contextually irrelevant**: zodiac personality traits and random career guidance snippets
- The LLM tried to **incorporate irrelevant retrieved facts** into what should have been a simple summary
- Response mixed summary with **unsolicited new advice**, confusing the user
- The retrieved context **diluted the summarization task**

### Without Retrieval (RAG OFF)
- Response correctly focused on **summarizing the conversation history**
- Clean, organized, and directly addressed the user's request
- No extraneous information injected

### Verdict
**Retrieval hurt the response quality.** The intent classifier correctly identifies "summarize" as a no-retrieval intent, preventing this degradation in production. This demonstrates why **intent-aware RAG** (not always-on retrieval) is critical.

---

## Key Takeaways

1. **Intent-aware retrieval is essential.** Blindly retrieving on every query degrades responses for conversational/meta queries.
2. **Similarity threshold filtering prevents noise.** Setting a threshold of 0.35 (cosine similarity) ensures only relevant contexts are included.
3. **Context-window trimming controls cost.** Capping retrieved context at 2000 tokens prevents unnecessary token expenditure.
4. **The two-stage classifier (rules + LLM fallback) balances cost and accuracy.** Rule-based classification handles ~70% of queries at zero cost; LLM handles edge cases.

## Cost Analysis

| Component | Tokens per Request (approx) | Cost Impact |
|-----------|---------------------------|-------------|
| Intent classification (rule-based) | 0 | Free |
| Intent classification (LLM fallback) | ~200 | Occasional |
| Retrieval context | ~500-2000 | Controlled by trimming |
| Hindi translation | ~100 | Only for Hindi queries |
| Response generation | ~500-1000 | Always |
| Memory summarization | ~300 | Only when window exceeded |

**Total per request:** ~700-3500 tokens (depending on retrieval and language)
