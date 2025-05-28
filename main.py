import json
import click
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


# Target properties for embedding as specified in CLAUDE.md
TARGET_PROPERTIES = [
    "appearance", "birthDate", "birthPlace", "country", "deathDate", "description",
    "education", "ethnicity", "firstName", "idNumber", "lastName", "middleName",
    "motherName", "name", "nameSuffix", "nationality", "passportNumber", "political",
    "position", "religion", "secondName", "spokenLanguage", "taxNumber", "title", "weight"
]

# Property weights for similarity calculation
PROPERTY_WEIGHTS = {
    "name": 0.3,
    "description": 0.2,
    "nationality": 0.15,
    "birthPlace": 0.1,
    "education": 0.1,
    "political": 0.05,
    "appearance": 0.05,
    # Other properties share remaining 0.05
    "firstName": 0.01,
    "lastName": 0.01,
    "country": 0.01,
    "birthDate": 0.01,
    "position": 0.01
}


def load_entities(json_file: Path) -> List[Dict[str, Any]]:
    """Load FTM entities from JSON file."""
    entities = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entity = json.loads(line)
                entities.append(entity)
    return entities


def preprocess_property(value: Any) -> Optional[str]:
    """Clean and preprocess property values for embedding."""
    if value is None:
        return None
    
    if isinstance(value, list):
        if not value:
            return None
        value = value[0] if len(value) == 1 else " | ".join(str(v) for v in value)
    
    value_str = str(value).strip()
    return value_str if value_str else None


def extract_entity_properties(entity: Dict[str, Any]) -> Dict[str, str]:
    """Extract and preprocess target properties from an entity."""
    properties = entity.get("properties", {})
    extracted = {}
    
    for prop in TARGET_PROPERTIES:
        if prop in properties:
            processed_value = preprocess_property(properties[prop])
            if processed_value:
                extracted[prop] = processed_value
    
    return extracted


def embed_properties(properties: Dict[str, str], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """Generate embeddings for entity properties."""
    embeddings = {}
    
    for prop, value in properties.items():
        try:
            embedding = model.encode(value, convert_to_numpy=True)
            embeddings[prop] = embedding
        except Exception as e:
            click.echo(f"Warning: Failed to embed property '{prop}': {e}")
            continue
    
    return embeddings


@click.group()
def cli():
    """FTM Entity Similarity Tool"""
    pass


@cli.command()
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Sentence transformer model to use",
)
@click.option(
    "--output-dir",
    default="./embeddings",
    help="Directory to save embeddings and index",
)
def embed(json_file: Path, model: str, output_dir: str):
    """Embed entity properties for similarity search."""
    click.echo(f"Loading entities from {json_file}")
    entities = load_entities(json_file)
    click.echo(f"Loaded {len(entities)} entities")
    
    click.echo(f"Loading sentence transformer model: {model}")
    sentence_model = SentenceTransformer(model)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store entity embeddings and metadata
    entity_embeddings = {}
    entity_metadata = {}
    
    click.echo("Processing entities and generating embeddings...")
    with click.progressbar(entities) as bar:
        for entity in bar:
            entity_id = entity.get("id")
            if not entity_id:
                continue
                
            # Extract and preprocess properties
            properties = extract_entity_properties(entity)
            if not properties:
                continue
                
            # Generate embeddings for each property
            embeddings = embed_properties(properties, sentence_model)
            if embeddings:
                entity_embeddings[entity_id] = embeddings
                entity_metadata[entity_id] = {
                    "original_data": entity,
                    "properties": properties
                }
    
    click.echo(f"Generated embeddings for {len(entity_embeddings)} entities")
    
    # Save embeddings and metadata
    embeddings_file = output_path / "embeddings.pkl"
    metadata_file = output_path / "metadata.pkl"
    weights_file = output_path / "weights.pkl"
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(entity_embeddings, f)
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(entity_metadata, f)
        
    with open(weights_file, 'wb') as f:
        pickle.dump(PROPERTY_WEIGHTS, f)
    
    click.echo(f"Saved embeddings to {embeddings_file}")
    click.echo(f"Saved metadata to {metadata_file}")
    click.echo(f"Saved weights to {weights_file}")
    
    # Print some statistics
    property_counts = {}
    for embeddings in entity_embeddings.values():
        for prop in embeddings.keys():
            property_counts[prop] = property_counts.get(prop, 0) + 1
    
    click.echo("\nProperty coverage:")
    for prop, count in sorted(property_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(entity_embeddings)) * 100
        click.echo(f"  {prop}: {count} entities ({percentage:.1f}%)")


@cli.command()
@click.argument("person_id")
@click.option(
    "--embeddings-dir",
    default="./embeddings",
    help="Directory containing embeddings and index",
)
@click.option("--top-k", default=5, help="Number of similar entities to return")
def search(person_id: str, embeddings_dir: str, top_k: int):
    pass


if __name__ == "__main__":
    cli()
