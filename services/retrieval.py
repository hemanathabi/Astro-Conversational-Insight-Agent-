import chromadb
import config
from services.llm_service import get_embeddings


def get_collection():
    """Get the ChromaDB collection handle."""
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return client.get_collection(
        name=config.CHROMA_COLLECTION_NAME,
    )


def retrieve_context(query, topic=None, zodiac=None, n_results=None):
    """Retrieve relevant context from ChromaDB with optional metadata filters.

    Args:
        query: The search query string
        topic: Optional topic filter (career, love, spiritual, planetary, personality, nakshatra)
        zodiac: Optional zodiac sign filter
        n_results: Number of results to fetch (default from config)

    Returns:
        List of dicts with content, source, score, and metadata
    """
    n_results = n_results or config.TOP_K_RESULTS
    collection = get_collection()
    embeddings = get_embeddings()

    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Build a list of queries to run (topic filter, zodiac filter, or both)
    # We run separate queries because not all docs have all metadata fields
    queries_to_run = []
    if topic:
        queries_to_run.append({"topic": {"$eq": topic}})
    if zodiac:
        queries_to_run.append({"zodiac": {"$eq": zodiac}})
    if not queries_to_run:
        queries_to_run.append(None)  # No filter — semantic search only

    # Collect results from all queries, dedup by document content
    seen_docs = set()
    contexts = []

    for where_filter in queries_to_run:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            continue

        if not results or not results["documents"] or not results["documents"][0]:
            continue

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if doc in seen_docs:
                continue
            seen_docs.add(doc)

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            similarity = 1 - (dist / 2)

            if similarity >= config.SIMILARITY_THRESHOLD:
                contexts.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "topic": meta.get("topic", "general"),
                    "score": round(similarity, 4),
                    "metadata": meta,
                })

    # Sort by score descending and limit to top n_results
    contexts.sort(key=lambda x: x["score"], reverse=True)
    return contexts[:n_results]


def trim_context(contexts, max_tokens=None):
    """Trim retrieved contexts to fit within token budget.

    Uses approximate token count (1 token ~= 4 chars) and drops
    lowest-scored results first.

    Args:
        contexts: List of context dicts (sorted by score descending)
        max_tokens: Maximum token budget (default from config)

    Returns:
        Trimmed list of contexts within token budget
    """
    max_tokens = max_tokens or config.MAX_CONTEXT_TOKENS
    trimmed = []
    total_tokens = 0

    for ctx in contexts:
        # Approximate token count
        token_estimate = len(ctx["content"]) // 4
        if total_tokens + token_estimate <= max_tokens:
            trimmed.append(ctx)
            total_tokens += token_estimate
        else:
            break

    return trimmed
