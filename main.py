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


def load_embeddings_data(embeddings_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load embeddings, metadata, and weights from files."""
    embeddings_path = Path(embeddings_dir)
    
    embeddings_file = embeddings_path / "embeddings.pkl"
    metadata_file = embeddings_path / "metadata.pkl"
    weights_file = embeddings_path / "weights.pkl"
    
    if not embeddings_file.exists():
        raise click.ClickException(f"Embeddings file not found: {embeddings_file}")
    if not metadata_file.exists():
        raise click.ClickException(f"Metadata file not found: {metadata_file}")
    if not weights_file.exists():
        raise click.ClickException(f"Weights file not found: {weights_file}")
    
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
        
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
    
    return embeddings, metadata, weights


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_weighted_similarity(
    query_embeddings: Dict[str, np.ndarray],
    candidate_embeddings: Dict[str, np.ndarray],
    weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """Calculate weighted similarity between query and candidate entity."""
    total_similarity = 0.0
    total_weight = 0.0
    property_similarities = {}
    
    # Find common properties
    common_properties = set(query_embeddings.keys()) & set(candidate_embeddings.keys())
    
    for prop in common_properties:
        if prop in weights:
            similarity = cosine_similarity(query_embeddings[prop], candidate_embeddings[prop])
            weight = weights[prop]
            
            total_similarity += weight * similarity
            total_weight += weight
            property_similarities[prop] = similarity
    
    # Normalize by actual total weight to handle missing properties
    if total_weight > 0:
        weighted_similarity = total_similarity / total_weight
    else:
        weighted_similarity = 0.0
    
    return weighted_similarity, property_similarities


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
@click.option("--show-details", is_flag=True, help="Show detailed similarity breakdown")
def search(person_id: str, embeddings_dir: str, top_k: int, show_details: bool):
    """Search for entities similar to the given person_id."""
    try:
        # Load embeddings, metadata, and weights
        click.echo(f"Loading embeddings from {embeddings_dir}")
        embeddings, metadata, weights = load_embeddings_data(embeddings_dir)
        
        # Check if query entity exists
        if person_id not in embeddings:
            available_ids = list(embeddings.keys())[:10]  # Show first 10 IDs
            click.echo(f"Entity '{person_id}' not found in embeddings.")
            click.echo(f"Available entity IDs (first 10): {available_ids}")
            if len(embeddings) > 10:
                click.echo(f"... and {len(embeddings) - 10} more")
            return
        
        query_embeddings = embeddings[person_id]
        query_metadata = metadata[person_id]
        
        click.echo(f"Searching for entities similar to '{person_id}'")
        click.echo(f"Query entity has {len(query_embeddings)} embedded properties")
        
        # Calculate similarities with all other entities
        similarities = []
        
        with click.progressbar(embeddings.items(), label="Calculating similarities") as bar:
            for entity_id, candidate_embeddings in bar:
                if entity_id == person_id:
                    continue  # Skip self
                
                weighted_sim, prop_similarities = calculate_weighted_similarity(
                    query_embeddings, candidate_embeddings, weights
                )
                
                similarities.append({
                    'entity_id': entity_id,
                    'similarity': weighted_sim,
                    'property_similarities': prop_similarities,
                    'common_properties': len(prop_similarities)
                })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Display results
        click.echo(f"\nTop {top_k} most similar entities to '{person_id}':")
        click.echo("=" * 60)
        
        # Show query entity info
        query_props = query_metadata.get('properties', {})
        click.echo(f"\nQuery Entity: {person_id}")
        click.echo(f"Name: {query_props.get('name', 'N/A')}")
        click.echo(f"Description: {query_props.get('description', 'N/A')[:100]}{'...' if len(query_props.get('description', '')) > 100 else ''}")
        click.echo(f"Nationality: {query_props.get('nationality', 'N/A')}")
        click.echo("")
        
        for i, result in enumerate(similarities[:top_k], 1):
            entity_id = result['entity_id']
            similarity = result['similarity']
            common_props = result['common_properties']
            
            candidate_metadata = metadata[entity_id]
            candidate_props = candidate_metadata.get('properties', {})
            
            click.echo(f"{i}. Entity ID: {entity_id}")
            click.echo(f"   Similarity Score: {similarity:.4f}")
            click.echo(f"   Common Properties: {common_props}")
            click.echo(f"   Name: {candidate_props.get('name', 'N/A')}")
            click.echo(f"   Description: {candidate_props.get('description', 'N/A')[:100]}{'...' if len(candidate_props.get('description', '')) > 100 else ''}")
            click.echo(f"   Nationality: {candidate_props.get('nationality', 'N/A')}")
            
            if show_details:
                click.echo("   Property Similarities:")
                prop_sims = result['property_similarities']
                for prop, sim in sorted(prop_sims.items(), key=lambda x: x[1], reverse=True):
                    weight = weights.get(prop, 0.0)
                    click.echo(f"     {prop}: {sim:.4f} (weight: {weight:.3f})")
            
            click.echo("")
            
    except Exception as e:
        raise click.ClickException(f"Search failed: {e}")


if __name__ == "__main__":
    cli()
