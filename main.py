import json
import click
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Any, Tuple


def preprocess_entity(entity: Dict[str, Any]) -> str:
    """Convert FTM entity to a document string for embedding."""
    parts = []
    
    # Key textual properties as specified in CLAUDE.md
    properties = [
        'appearance', 'birthDate', 'birthPlace', 'country', 'deathDate', 
        'description', 'education', 'ethnicity', 'firstName', 'idNumber', 
        'lastName', 'middleName', 'motherName', 'name', 'nameSuffix', 
        'nationality', 'passportNumber', 'political', 'position', 'religion', 
        'secondName', 'spokenLanguage', 'taxNumber', 'title', 'weight'
    ]
    
    for prop in properties:
        value = entity.get(prop)
        if value:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        parts.append(f"{prop}: {item}")
            elif isinstance(value, str) and value.strip():
                parts.append(f"{prop}: {value}")
            elif value is not None:  # Handle non-string values like numbers
                parts.append(f"{prop}: {str(value)}")
    
    return " | ".join(parts) if parts else f"Entity {entity.get('id', 'unknown')}"


def create_embeddings(entities: List[Dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[np.ndarray, SentenceTransformer]:
    """Generate embeddings for all entities."""
    model = SentenceTransformer(model_name)
    
    documents = [preprocess_entity(entity) for entity in entities]
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    return embeddings, model


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create and populate FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index


@click.command()
@click.argument('json_file', type=click.Path(exists=True, path_type=Path))
@click.option('--model', default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence transformer model to use")
@click.option('--output-dir', default="./embeddings", help="Directory to save embeddings and index")
def embed_entities(json_file: Path, model: str, output_dir: str):
    """Load FTM entities from JSONL file and create embeddings with FAISS index."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    click.echo(f"Loading entities from {json_file}")
    entities = []
    with open(json_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entities.append(json.loads(line))
    
    click.echo(f"Processing {len(entities)} entities...")
    
    # Create embeddings
    embeddings, transformer_model = create_embeddings(entities, model)
    click.echo(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    click.echo(f"Created FAISS index with {index.ntotal} vectors")
    
    # Save everything
    entities_file = output_path / "entities.json"
    embeddings_file = output_path / "embeddings.npy"
    index_file = output_path / "faiss_index.bin"
    
    with open(entities_file, 'w') as f:
        json.dump(entities, f, indent=2)
    
    np.save(embeddings_file, embeddings)
    faiss.write_index(index, str(index_file))
    
    click.echo(f"Saved entities to: {entities_file}")
    click.echo(f"Saved embeddings to: {embeddings_file}")
    click.echo(f"Saved FAISS index to: {index_file}")
    click.echo(f"Vector store created successfully with {len(entities)} entities!")


if __name__ == "__main__":
    embed_entities()
