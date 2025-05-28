# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project called "ftm-similarity" built with modern Python packaging (pyproject.toml).

**Project Goal:** Create a Proof of Concept (POC) for calculating similarity between FollowTheMoney (FTM) entities using property-based embeddings with weighted averaging.

**Input:** A JSON file containing 400 FTM entities.

**Output:** A Python script that demonstrates preprocessing, embedding individual properties, storing, and querying for similar entities using weighted similarity scores.

**Core Steps for the Coding Model:**

1.  **Load Data:**

    - Read the FTM entities from the provided JSON file.

2.  **Property-Based Preprocessing:**

    - **Objective:** Instead of creating single document strings, process each property individually for separate embedding.
    - **Target Properties:** appearance, birthDate, birthPlace, country, deathDate, description, education, ethnicity, firstName, idNumber, lastName, middleName, motherName, name, nameSuffix, nationality, passportNumber, political, position, religion, secondName, spokenLanguage, taxNumber, title, weight
    - **Process:**
      - For each entity, extract and clean individual property values
      - Handle missing or `None` values gracefully (e.g., skip property or use empty string)
      - Group properties by type (e.g., textual vs. categorical vs. date-based)

3.  **Property-Specific Embedding:**

    - **Model:** Use a pre-trained sentence transformer model (e.g., `sentence-transformers/all-MiniLM-L6-v2` or similar)
    - **Process:**
      - Generate separate embedding vectors for each property that has a meaningful value
      - For textual properties (description, appearance, education): embed directly
      - For categorical properties (country, nationality, political): embed as-is or with context
      - For names (firstName, lastName, name): embed individually
      - Store embeddings in a structured format: `{entity_id: {property_name: embedding_vector}}`

4.  **Weighted Similarity Configuration:**

    - **Property Weights:** Define importance weights for different properties, e.g.:
      - High importance: name (0.3), description (0.2), nationality (0.15)
      - Medium importance: birthPlace (0.1), education (0.1), political (0.05)
      - Lower importance: appearance (0.05), other properties (0.05 total)
    - **Weights should sum to 1.0**
    - Make weights configurable/adjustable

5.  **Vector Storage:**

    - **Choice:** Use an in-memory structure to store property-specific embeddings
    - **Structure:** Dictionary or class-based approach: `{entity_id: {property: embedding, ...}, original_data: {...}}`
    - **Optional:** Use FAISS for each property type if performance optimization is needed

6.  **Weighted Similarity Query:**
    - **Selection:** Given an arbitrary entity from the dataset to use as a query
    - **Property Embedding:** Embed each property of the query entity using the same models/logic as step 3
    - **Similarity Calculation:**
      - For each candidate entity, calculate cosine similarity between corresponding property embeddings
      - Apply weighted average: `total_similarity = Σ(weight_i × similarity_i)` for all matching properties
      - Handle cases where properties are missing in either query or candidate entity
    - **Search:** Rank all entities by their weighted similarity scores
    - **Result:** Return the top N most similar entities (e.g., Top 5 or Top 10), including:
      - Original entity data or ID
      - Overall weighted similarity score
      - Breakdown of individual property similarities (optional, for debugging)

**Constraints & Best Practices:**

- Handle common errors (e.g., file not found, missing properties)
- Focus on clear, modular code with separate functions for each property type
- Make the weighting system easily configurable
- Avoid external databases unless explicitly simple setup instructions can be provided
- No need for production-ready features (logging, extensive error handling, etc.)
- Consider normalizing similarity scores to [0, 1] range for interpretability

## Development Commands

Don't run development commands. Don't install dependencies. Instead, let the
user know what packages you need and what commands can be run.

## Architecture

- **main.py**: Entry point containing the main() function
- **pyproject.toml**: Project configuration and dependencies (currently minimal)
- Python 3.12+ required

## Example of the input data

```
{"id": "person1", "schema": "Person", "properties": {"name": "John Smith", "country": "US", "nationality": "US", "birthDate": "1975-06-15", "birthPlace": "Boston, MA", "idNumber": "SSN-123-45-6789", "education": "Harvard University 1997", "political": "Democratic Party", "appearance": "Tall, athletic build, salt-and-pepper hair", "description": "Former investment banker who transitioned to politics"}}
{"id": "person2", "schema": "Person", "properties": {"firstName": "Jean", "lastName": "Dupont", "country": "FR", "nationality": "FR", "birthDate": "1982-03-22", "passportNumber": "FR12345678", "education": "Sciences Po Paris 2004", "political": "La République En Marche", "appearance": "Well-dressed, distinguished appearance, always in tailored suits", "description": "Tech entrepreneur who founded successful AI startup"}}
```
