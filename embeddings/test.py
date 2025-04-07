import torch
import json
from embeddings.embeddings_client import TogetherEmbeddingsClient
from .vector_db import QdrantManager
from qdrant_client.http import models
from typing import Dict, Any, Optional

# Constants
QDRANT_COLLECTION_NAME = "shl_assessments_with_metadata"
MODEL_CACHE_DIR = "/Users/akshat/Developer/Tasks/SHL/.model_cache"

def initialize_embeddings():
    """Initialize the Together AI client for embeddings."""
    return TogetherEmbeddingsClient()

# def initialize_embeddings():
#     """Initialize the embedding model - must use the same model as in the generator."""
#     model_name = "BAAI/bge-large-en-v1.5"
    
#     # # Check for available hardware acceleration
#     # if torch.cuda.is_available():
#     #     device = "cuda"
#     #     print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
#     # elif torch.backends.mps.is_available():
#     #     device = "mps"
#     #     print("Using MPS (Metal Performance Shaders) on macOS.")
#     # else:
#     #     device = "cpu"
#     #     print("GPU acceleration not available, using CPU. This will be slower.")
#     device = "cpu"
    
#     model_kwargs = {'device': device}
#     encode_kwargs = {'normalize_embeddings': True}
    
#     print(f"Loading model: {model_name} on {device}")
    
#     embeddings = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs,
#         cache_folder=MODEL_CACHE_DIR
#     )
#     return embeddings

def search_assessment_db(query_text, embeddings, qdrant_manager, filters=None, top_k=5):
    """Search the vector DB for the top k results matching the query."""
    print(f"Searching for: '{query_text}'")
    
    # For BGE models, adding an instruction prefix improves search quality
    instruction = "Represent this sentence for searching relevant passages: "
    query_with_instruction = f"{instruction}{query_text}"
    
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query_with_instruction)
    
    # Prepare filter as a dictionary if provided
    search_filter = None
    if filters:
        # Convert the Filter object to a dictionary representation for the search method
        filter_obj = create_qdrant_filter(filters)
        # Create a dictionary representation of the filter
        if filter_obj and hasattr(filter_obj, "must"):
            search_filter = {"must": filter_obj.must}
        if filter_obj and hasattr(filter_obj, "should"):
            search_filter = search_filter or {}
            search_filter["should"] = filter_obj.should
        print(f"Applying filters: {json.dumps(filters, indent=2)}")
    
    # Search Qdrant with the query embedding
    results = qdrant_manager.search(
        query_vector=query_embedding,
        filter_conditions=search_filter,
        limit=top_k
    )
    
    return results

def create_qdrant_filter(filters: Dict[str, Any]) -> Optional[models.Filter]:
    """
    Create a Qdrant filter from the provided filter dictionary.
    
    Args:
        filters: Dictionary containing filter conditions.
        
    Returns:
        Qdrant Filter object or None if no filters are provided.
    """
    must_conditions = []
    
    for field, value in filters.items():
        if isinstance(value, list):
            # Ensure the value is a list for MatchAny
            should_conditions = [
                models.FieldCondition(
                    key=field,
                    match=models.MatchAny(any=[item])  # Wrap each item in a list
                )
                for item in value
            ]
            if should_conditions:
                must_conditions.append(models.Filter(should=should_conditions))
        elif field == "duration_minutes":
            # Handle range conditions for duration
            if "min" in value:
                must_conditions.append(
                    models.FieldCondition(
                        key="duration_minutes",
                        range=models.Range(gte=value["min"])
                    )
                )
            if "max" in value:
                must_conditions.append(
                    models.FieldCondition(
                        key="duration_minutes",
                        range=models.Range(lte=value["max"])
                    )
                )
        else:
            # Handle single-value fields
            must_conditions.append(
                models.FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value)
                )
            )
    
    if must_conditions:
        return models.Filter(must=must_conditions)
    return None

def display_search_results(results):
    """Display the search results in a readable format."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:")
    print("-" * 100)
    
    for i, result in enumerate(results):
        score = result.score
        payload = result.payload
        
        # Extract key information from payload
        assessment_name = payload.get("assessment_name", "Unknown Assessment")
        content_type = payload.get("content_type", "Unknown Type")
        
        # Print the result with similarity score (formatted as percentage)
        print(f"{i+1}. {assessment_name} - Similarity: {score:.2%}")
        print(f"   Content Type: {content_type}")
        
        # Print all other metadata fields
        skipped_fields = ["assessment_name", "content_type", "text", "doc_id"]
        
        for key, value in sorted(payload.items()):
            if key in skipped_fields:
                continue
                
            # Format the output based on the field type
            if isinstance(value, list):
                if value and all(isinstance(item, str) for item in value):
                    print(f"   {key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            elif key == "duration_minutes" and value is not None:
                print(f"   Duration: {value} minutes")
            elif isinstance(value, dict):
                print(f"   {key.replace('_', ' ').title()}: {json.dumps(value)}")
            elif value is not None and value != "":
                print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Extract description from text (usually after the assessment name)
        if "text" in payload:
            text_parts = payload["text"].split("\n", 1)
            if len(text_parts) > 1:
                description = text_parts[1]
                # Truncate description if too long
                if len(description) > 300:
                    description = description[:297] + "..."
                print(f"   Description: {description}")
        
        print("-" * 100)

def get_filter_options():
    """Interactive interface for creating search filters"""
    filters = {}
    print("\n--- Filter Options ---")
    print("Enter values for the filters you want to apply (leave empty to skip):")
    
    # Job Levels filter
    job_levels = input("Job Levels (comma-separated, e.g., Manager, Graduate): ")
    if job_levels.strip():
        filters["job_levels"] = [level.strip() for level in job_levels.split(",")]
    
    # Languages filter
    languages = input("Languages (comma-separated, e.g., English, French): ")
    if languages.strip():
        filters["languages"] = [lang.strip() for lang in languages.split(",")]
    
    # Test Type Categories filter
    test_types = input("Test Categories (comma-separated, e.g., Ability and Aptitude, Knowledge and Skills): ")
    if test_types.strip():
        filters["test_type_categories"] = [tt.strip() for tt in test_types.split(",")]
    
    # Remote testing filter
    remote = input("Remote Testing (Yes/No): ")
    if remote.strip().lower() in ["yes", "y"]:
        filters["remote_testing"] = "Yes"
    elif remote.strip().lower() in ["no", "n"]:
        filters["remote_testing"] = "No"
    
    # Duration filter
    min_duration = input("Minimum Duration (in minutes): ")
    max_duration = input("Maximum Duration (in minutes): ")
    if min_duration.strip() or max_duration.strip():
        filters["duration_minutes"] = {}
        if min_duration.strip() and min_duration.isdigit():
            filters["duration_minutes"]["min"] = int(min_duration)
        if max_duration.strip() and max_duration.isdigit():
            filters["duration_minutes"]["max"] = int(max_duration)
    
    return filters if filters else None

def main():
    """Main function to run the search query."""
    print("Initializing embedding model...")
    embeddings = initialize_embeddings()
    
    # Get the embedding dimension
    vector_size = 1024  # For BGE-large-en-v1.5
    
    # Initialize QdrantManager with the same collection name
    qdrant_manager = QdrantManager(
        collection_name=QDRANT_COLLECTION_NAME, 
        vector_size=vector_size
    )
    
    # Process queries until user exits
    while True:
        print("\n=== SHL Assessment Search ===")
        print("1. Simple Search")
        print("2. Advanced Search (with filters)")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "3":
            print("Exiting...")
            break
        
        # Get query from user
        query = input("\nEnter your search query: ")
        
        if query.lower() == 'exit':
            break
        
        # Apply filters for advanced search
        filters = None
        if choice == "2":
            filters = get_filter_options()
        
        # Set number of results to display
        try:
            top_k = int(input("\nHow many results do you want to see? (default: 5): ") or "5")
        except ValueError:
            top_k = 5
            print("Invalid input, using default value: 5")
        
        # Search for similar assessments
        results = search_assessment_db(query, embeddings, qdrant_manager, filters, top_k)
        
        # Display results
        display_search_results(results)

if __name__ == "__main__":
    main()