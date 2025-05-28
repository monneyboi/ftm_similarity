import json
import click
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Any, Tuple


def preprocess_entity(entity: Dict[str, Any]) -> str:
    """Convert FTM entity to a human-readable story for embedding."""
    story_parts = []
    
    # Extract key information
    name = _get_full_name(entity)
    birth_info = _get_birth_info(entity)
    death_info = _get_death_info(entity)
    location_info = _get_location_info(entity)
    professional_info = _get_professional_info(entity)
    personal_info = _get_personal_info(entity)
    identification_info = _get_identification_info(entity)
    
    # Build the story
    if name:
        story_parts.append(f"This is {name}.")
    
    if birth_info:
        story_parts.append(birth_info)
    
    if location_info:
        story_parts.append(location_info)
    
    if professional_info:
        story_parts.append(professional_info)
    
    if personal_info:
        story_parts.append(personal_info)
    
    if death_info:
        story_parts.append(death_info)
    
    if identification_info:
        story_parts.append(identification_info)
    
    # Add description if available
    properties = entity.get('properties', {})
    description = properties.get('description')
    if description:
        if isinstance(description, list):
            for desc in description:
                if isinstance(desc, str) and desc.strip():
                    story_parts.append(desc.strip())
        elif isinstance(description, str) and description.strip():
            story_parts.append(description.strip())
    
    return " ".join(story_parts) if story_parts else f"Unknown entity {entity.get('id', '')}"


def _get_full_name(entity: Dict[str, Any]) -> str:
    """Construct full name from name components."""
    properties = entity.get('properties', {})
    name_parts = []
    
    # Try main name first
    if properties.get('name'):
        return str(properties['name'])
    
    # Build from components
    title = properties.get('title')
    if title:
        name_parts.append(str(title))
    
    first_name = properties.get('firstName')
    if first_name:
        name_parts.append(str(first_name))
    
    middle_name = properties.get('middleName')
    if middle_name:
        name_parts.append(str(middle_name))
    
    second_name = properties.get('secondName')
    if second_name:
        name_parts.append(str(second_name))
    
    last_name = properties.get('lastName')
    if last_name:
        name_parts.append(str(last_name))
    
    suffix = properties.get('nameSuffix')
    if suffix:
        name_parts.append(str(suffix))
    
    return " ".join(name_parts) if name_parts else ""


def _get_birth_info(entity: Dict[str, Any]) -> str:
    """Create birth information sentence."""
    properties = entity.get('properties', {})
    birth_date = properties.get('birthDate')
    birth_place = properties.get('birthPlace')
    
    if birth_date and birth_place:
        return f"Born on {birth_date} in {birth_place}."
    elif birth_date:
        return f"Born on {birth_date}."
    elif birth_place:
        return f"Born in {birth_place}."
    
    return ""


def _get_death_info(entity: Dict[str, Any]) -> str:
    """Create death information sentence."""
    properties = entity.get('properties', {})
    death_date = properties.get('deathDate')
    if death_date:
        return f"Died on {death_date}."
    return ""


def _get_location_info(entity: Dict[str, Any]) -> str:
    """Create location and nationality information."""
    properties = entity.get('properties', {})
    parts = []
    
    nationality = properties.get('nationality')
    if nationality:
        if isinstance(nationality, list):
            nationalities = [str(n) for n in nationality if n]
            if nationalities:
                if len(nationalities) == 1:
                    parts.append(f"Holds {nationalities[0]} nationality.")
                else:
                    parts.append(f"Holds multiple nationalities: {', '.join(nationalities)}.")
        else:
            parts.append(f"Holds {nationality} nationality.")
    
    country = properties.get('country')
    if country and country != nationality:
        parts.append(f"Associated with {country}.")
    
    return " ".join(parts)


def _get_professional_info(entity: Dict[str, Any]) -> str:
    """Create professional information sentence."""
    properties = entity.get('properties', {})
    parts = []
    
    position = properties.get('position')
    if position:
        if isinstance(position, list):
            positions = [str(p) for p in position if p]
            if positions:
                if len(positions) == 1:
                    parts.append(f"Works as {positions[0]}.")
                else:
                    parts.append(f"Has held positions including {', '.join(positions)}.")
        else:
            parts.append(f"Works as {position}.")
    
    political = properties.get('political')
    if political:
        parts.append(f"Has political affiliation with {political}.")
    
    education = properties.get('education')
    if education:
        if isinstance(education, list):
            edu_list = [str(e) for e in education if e]
            if edu_list:
                parts.append(f"Education includes {', '.join(edu_list)}.")
        else:
            parts.append(f"Education includes {education}.")
    
    return " ".join(parts)


def _get_personal_info(entity: Dict[str, Any]) -> str:
    """Create personal characteristics information."""
    properties = entity.get('properties', {})
    parts = []
    
    ethnicity = properties.get('ethnicity')
    if ethnicity:
        parts.append(f"Of {ethnicity} ethnicity.")
    
    religion = properties.get('religion')
    if religion:
        parts.append(f"Practices {religion}.")
    
    spoken_language = properties.get('spokenLanguage')
    if spoken_language:
        if isinstance(spoken_language, list):
            languages = [str(l) for l in spoken_language if l]
            if languages:
                if len(languages) == 1:
                    parts.append(f"Speaks {languages[0]}.")
                else:
                    parts.append(f"Speaks {', '.join(languages)}.")
        else:
            parts.append(f"Speaks {spoken_language}.")
    
    mother_name = properties.get('motherName')
    if mother_name:
        parts.append(f"Mother's name is {mother_name}.")
    
    appearance = properties.get('appearance')
    if appearance:
        parts.append(f"Physical appearance: {appearance}.")
    
    weight = properties.get('weight')
    if weight:
        parts.append(f"Weight: {weight}.")
    
    return " ".join(parts)


def _get_identification_info(entity: Dict[str, Any]) -> str:
    """Create identification information with full details."""
    properties = entity.get('properties', {})
    parts = []
    
    id_number = properties.get('idNumber')
    if id_number:
        parts.append(f"Official ID number: {id_number}.")
    
    passport_number = properties.get('passportNumber')
    if passport_number:
        parts.append(f"Passport number: {passport_number}.")
    
    tax_number = properties.get('taxNumber')
    if tax_number:
        parts.append(f"Tax identification number: {tax_number}.")
    
    return " ".join(parts)


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
