"""
Knowledge Base Ingestion Script
Embeds all data files into ChromaDB with metadata tags for filtered retrieval.

Run this script once before starting the Flask app:
    python knowledge/ingest.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import config
from services.llm_service import get_embeddings


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def build_documents():
    """Build document list with content, metadata, and IDs for ChromaDB."""
    documents = []
    metadatas = []
    ids = []

    # 1. Zodiac traits — one document per zodiac sign
    zodiac_data = load_json(os.path.join(config.DATA_DIR, "zodiac_traits.json"))
    for sign, traits in zodiac_data.items():
        content = (
            f"Zodiac Sign: {sign}. "
            f"Element: {traits['element']}. "
            f"Ruling Planet: {traits['ruling_planet']}. "
            f"Personality: {traits['personality']} "
            f"Strengths: {traits['strengths']} "
            f"Challenges: {traits['challenges']}"
        )
        documents.append(content)
        metadatas.append({
            "source": "zodiac_traits",
            "topic": "personality",
            "zodiac": sign,
            "element": traits["element"],
        })
        ids.append(f"zodiac_{sign.lower()}")

    # 2. Planetary impacts — one document per planet
    planetary_data = load_json(os.path.join(config.DATA_DIR, "planetary_impacts.json"))
    for planet, info in planetary_data.items():
        affects_str = ", ".join(info["affects"])
        content = (
            f"Planet: {planet} ({info['vedic_name']}). "
            f"Nature: {info['nature']}. "
            f"Description: {info['description']} "
            f"Affects: {affects_str}. "
            f"Remedies: {info['remedies']}"
        )
        if info.get("zodiac_rulership"):
            content += f" Rules: {info['zodiac_rulership']}."

        documents.append(content)
        metadatas.append({
            "source": "planetary_impacts",
            "topic": "planetary",
            "planet": planet,
            "nature": info["nature"],
        })
        ids.append(f"planet_{planet.lower()}")

    # 3. Career guidance — one document per line
    career_lines = load_text(os.path.join(config.DATA_DIR, "career_guidance.txt"))
    for i, line in enumerate(career_lines):
        documents.append(f"Career Guidance: {line}")
        metadatas.append({
            "source": "career_guidance",
            "topic": "career",
        })
        ids.append(f"career_{i}")

    # 4. Love guidance — one document per line
    love_lines = load_text(os.path.join(config.DATA_DIR, "love_guidance.txt"))
    for i, line in enumerate(love_lines):
        documents.append(f"Love & Relationship Guidance: {line}")
        metadatas.append({
            "source": "love_guidance",
            "topic": "love",
        })
        ids.append(f"love_{i}")

    # 5. Spiritual guidance — one document per line
    spiritual_lines = load_text(os.path.join(config.DATA_DIR, "spiritual_guidance.txt"))
    for i, line in enumerate(spiritual_lines):
        documents.append(f"Spiritual Guidance: {line}")
        metadatas.append({
            "source": "spiritual_guidance",
            "topic": "spiritual",
        })
        ids.append(f"spiritual_{i}")

    # 6. Nakshatra mapping — one document per nakshatra
    nakshatra_data = load_json(os.path.join(config.DATA_DIR, "nakshatra_mapping.json"))
    for name, info in nakshatra_data.items():
        content = (
            f"Nakshatra: {name}. "
            f"Zodiac: {info['zodiac_sign']}. "
            f"Ruling Planet: {info['ruling_planet']}. "
            f"Degree Range: {info['degree_range']}. "
            f"Deity: {info['deity']}. "
            f"Description: {info['description']} "
            f"Qualities: {info['qualities']}"
        )
        documents.append(content)
        metadatas.append({
            "source": "nakshatra_mapping",
            "topic": "nakshatra",
            "zodiac": info["zodiac_sign"].split("/")[0],  # Use primary sign
            "ruling_planet": info["ruling_planet"],
        })
        ids.append(f"nakshatra_{name.lower().replace(' ', '_')}")

    return documents, metadatas, ids


def ingest():
    """Embed and store all documents in ChromaDB."""
    print("Building documents from data files...")
    documents, metadatas, ids = build_documents()
    print(f"  Total documents: {len(documents)}")

    print(f"Initializing ChromaDB at {config.CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    # Delete existing collection if it exists (for re-ingestion)
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        print("  Deleted existing collection.")
    except Exception:
        pass

    embeddings = get_embeddings()

    print("Generating embeddings (this may take a minute)...")
    # Batch embed all documents
    embedding_vectors = embeddings.embed_documents(documents)
    print(f"  Generated {len(embedding_vectors)} embeddings.")

    print("Creating ChromaDB collection and adding documents...")
    collection = client.create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            embeddings=embedding_vectors[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"  Added batch {i // batch_size + 1} ({end}/{len(documents)} documents)")

    print(f"\nIngestion complete! {collection.count()} documents stored in ChromaDB.")
    return collection


if __name__ == "__main__":
    ingest()
