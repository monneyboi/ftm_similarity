# FTM Similarity

A proof of concept for calculating similarity between FollowTheMoney (FTM) entities using embeddings.

## Overview

This project demonstrates how to:
- Load FTM entities from JSONL files
- Preprocess entities into human-readable stories for embedding
- Generate embeddings using sentence transformers
- Store vectors in FAISS for efficient similarity search
- Query for similar entities

## Commands

### embed

Load FTM entities from JSONL file and create embeddings with FAISS index.

```bash
python main.py embed <json_file> [OPTIONS]
```

**Arguments:**
- `json_file`: Path to JSONL file containing FTM entities

**Options:**
- `--model`: Sentence transformer model to use (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--output-dir`: Directory to save embeddings and index (default: `./embeddings`)

**Example:**
```bash
python main.py embed entities.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --output-dir ./embeddings
```

**Output:**
- `entities.json`: Original entity data
- `stories.json`: Preprocessed entity stories
- `embeddings.npy`: Entity embedding vectors
- `faiss_index.bin`: FAISS similarity search index

### search

Search for entities similar to a given person ID.

```bash
python main.py search <person_id> [OPTIONS]
```

**Arguments:**
- `person_id`: ID of the entity to find similar entities for

**Options:**
- `--embeddings-dir`: Directory containing embeddings and index (default: `./embeddings`)
- `--top-k`: Number of similar entities to return (default: 5)

**Example:**
```bash
python main.py search person1 --top-k 10 --embeddings-dir ./embeddings
```

**Output:**
Displays the top-k most similar entities with:
- Entity ID and similarity score
- Name and key details (birth date, nationality, position)
- Preprocessed story excerpt

## Requirements

- Python 3.12+
- Dependencies: `click`, `sentence-transformers`, `faiss-cpu`, `numpy`

## Input Format

JSONL file with one FTM entity per line:

```json
{"id": "person1", "schema": "Person", "properties": {"name": "John Smith", "country": "US", "nationality": "US", "birthDate": "1975-06-15", "birthPlace": "Boston, MA", "description": "Former investment banker"}}
{"id": "person2", "schema": "Person", "properties": {"firstName": "Jean", "lastName": "Dupont", "country": "FR", "nationality": "FR", "birthDate": "1982-03-22", "description": "Tech entrepreneur"}}
```

## Workflow

1. **Prepare data**: Create JSONL file with FTM entities
2. **Generate embeddings**: `python main.py embed entities.jsonl`
3. **Search similarities**: `python main.py search person1`