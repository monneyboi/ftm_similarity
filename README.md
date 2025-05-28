# FTM Similarity

A proof of concept for calculating similarity between FollowTheMoney (FTM) entities using property-based embeddings with weighted averaging.

## Overview

This project demonstrates how to:
- Load FTM entities from JSONL files
- Extract and preprocess individual entity properties for separate embedding
- Generate property-specific embeddings using sentence transformers
- Calculate weighted similarity scores based on property importance
- Query for similar entities using weighted averaging of property similarities

## Commands

### embed

Load FTM entities from JSONL file and create property-based embeddings.

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
- `embeddings.pkl`: Property-specific embedding vectors for each entity
- `metadata.pkl`: Original entity data and preprocessed properties
- `weights.pkl`: Property importance weights for similarity calculation

**Target Properties:**
The system extracts and embeds 25 specific properties including name, description, nationality, birthPlace, education, political affiliation, appearance, and others. Missing composite properties (like 'name') are automatically constructed from component parts (firstName, lastName, etc.).

### search

Search for entities similar to a given person ID using weighted property similarity.

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
- Entity ID and weighted similarity score (0-1 scale)
- Name, description, and nationality
- Detailed breakdown of property-level similarities and their weighted contributions
- Number of common properties used in calculation

## Requirements

- Python 3.12+
- Dependencies: `click`, `sentence-transformers`, `numpy`

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