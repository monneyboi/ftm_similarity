# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project called "ftm-similarity" built with modern Python packaging (pyproject.toml).

**Project Goal:** Create a Proof of Concept (POC) for calculating similarity between FollowTheMoney (FTM) entities using embeddings.

**Input:** A JSON file containing 400 FTM entities.

**Output:** A Python script that demonstrates preprocessing, embedding, storing, and querying for similar entities.

**Core Steps for the Coding Model:**

1.  **Load Data:**

    - Read the FTM entities from the provided JSON file.

2.  **Entity Preprocessing (Document Creation for Embedding):**

    - **Objective:** Transform each structured FTM entity into a single, comprehensive string (a "document") suitable for a sentence transformer.
    - **Process:**

      - For each entity, concatenate key textual properties: appearance, birthDate, birthPlace, country, deathDate, description, education, ethnicity, firstName, idNumber, lastName, middleName, motherName, name, nameSuffix, nationality, passportNumber, political, position, religion, secondName, spokenLanguage, taxNumber, title, weight

      - Handle missing or `None` values gracefully (e.g., skip or use an empty string).

3.  **Entity Embedding:**

    - **Model:** Use a pre-trained sentence transformer model (e.g., `sentence-transformers/all-MiniLM-L6-v2` or a similar small, efficient model).
    - **Process:** Generate a fixed-size embedding vector for each preprocessed entity document string.

4.  **Vector Storage:**

    - **Choice:** Use an in-memory vector store. `FAISS` is recommended for efficient similarity search, or a simple list of `(entity_id, original_entity_data, embedding_vector)` tuples if FAISS is too complex for the time constraint.
    - **If FAISS:** Create a FAISS index (e.g., `IndexFlatL2` or `IndexFlatIP`) and add the embeddings.

5.  **Similarity Query:**
    - **Selection:** Given an arbitrary entity from the dataset to use as a query.
    - **Preprocessing:** Preprocess this query entity into its document string using the _exact same logic_ as step 2.
    - **Embedding:** Embed the query document using the _exact same model_ as step 3.
    - **Search:**
      - If using FAISS: Perform a k-nearest neighbors search (`index.search()`).
      - If using a list: Calculate cosine similarity between the query embedding and all stored embeddings.
    - **Result:** Return the top N most similar entities (e.g., Top 5 or Top 10), including their original data or ID and similarity score.

**Constraints & Best Practices:**

- Handle common errors (e.g., file not found).
- Focus on clear, modular code.
- Avoid external databases unless explicitly simple setup instructions can be provided (FAISS avoids this).
- No need for production-ready features (logging, extensive error handling, etc.).

## Development Commands

Don't run development commands. Don't install dependencies. Instead, let the
user know what packages you need and what commands can be run.

## Architecture

- **main.py**: Entry point containing the main() function
- **pyproject.toml**: Project configuration and dependencies (currently minimal)
- Python 3.12+ required
