import re
import json
import uuid
import hashlib
import torch
import os
from tqdm import tqdm
# Updated import to avoid deprecation warning
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings_client import TogetherEmbeddingsClient
from vector_db import QdrantManager

# File paths
INPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/processed_assessments_no_duplicates.json"
QDRANT_COLLECTION_NAME = "shl_assessments_with_metadata"
MODEL_CACHE_DIR = "/Users/akshat/Developer/Tasks/SHL/.model_cache"

# Ensure cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Test Type Keys mapping
TEST_TYPE_MAPPING = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations"
}

# def initialize_embeddings():
#     # model_name = "BAAI/bge-small-en-v1.5"  # Smaller, faster option (125M parameters)
#     model_name = "BAAI/bge-large-en-v1.5"    # Original large model (335M parameters)
    
#     # Check for available hardware acceleration
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
    
#     # Configure model - removed cache_folder from model_kwargs to avoid conflict
#     model_kwargs = {
#         'device': device
#     }
#     encode_kwargs = {'normalize_embeddings': True}
    
#     print(f"Loading model: {model_name} on {device}")
#     print(f"Model will be cached at: {MODEL_CACHE_DIR}")
    
#     embeddings = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs,
#         cache_folder=MODEL_CACHE_DIR  # Specify cache_folder as a separate parameter
#     )
#     return embeddings

def initialize_embeddings():
    """Initialize the Together AI client for embeddings."""
    return TogetherEmbeddingsClient()

# Get embedding dimension by testing a simple embedding
def get_embedding_dimension(embeddings_model):
    print("Determining embedding dimension...")
    # Generate an embedding for a simple text to determine the dimension
    sample_embedding = embeddings_model.embed_query("test")
    dimension = len(sample_embedding)
    print(f"Embedding dimension: {dimension}")
    return dimension

# Load data from JSON
def load_data(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Extract minutes from assessment length
def extract_minutes(assessment_length):
    if not assessment_length:
        return None
    
    match = re.search(r'(\d+)\s*minutes', assessment_length)
    if match:
        return int(match.group(1))
    
    match = re.search(r'(\d+)', assessment_length)
    if match:
        return int(match.group(1))
    
    return None

# Parse job levels into a list
def parse_job_levels(job_levels):
    if not job_levels:
        return []
    
    # Split by commas and clean up
    return [level.strip() for level in job_levels.split(',') if level.strip()]

# Parse languages into a list
def parse_languages(languages):
    if not languages:
        return []
    
    # Split by commas and clean up
    return [lang.strip() for lang in languages.split(',') if lang.strip()]

# Process test type keys
def process_test_types(test_type_keys):
    if not test_type_keys:
        return [], []
    
    # Extract key codes
    key_codes = [key.strip() for key in test_type_keys if key.strip()]
    
    # Map keys to their full descriptions
    key_descriptions = [TEST_TYPE_MAPPING.get(key, "Unknown") for key in key_codes]
    
    return key_codes, key_descriptions

# Generate a unique document ID
def generate_document_id(assessment_name, chunk_index=None):
    # Remove spaces and special characters from assessment name
    base_name = re.sub(r'[^\w]', '', assessment_name).lower()
    
    # If this is a chunk of a larger document, append the chunk index
    if chunk_index is not None:
        unique_id = f"{base_name}_chunk_{chunk_index}"
    else:
        unique_id = f"{base_name}"
    
    return unique_id

def extract_data_with_metadata(data):
    """Extract text and metadata for embeddings - One embedding per assessment."""
    documents = []
    
    for assessment in data:
        assessment_name = assessment.get("Assessment Name", "Unknown Assessment")
        
        # Process test type keys
        test_type_keys = assessment.get("Test Type Keys", [])
        key_codes, key_descriptions = process_test_types(test_type_keys)
        
        # Extract and process metadata
        metadata = {
            "assessment_name": assessment_name,
            "remote_testing": assessment.get("Remote Testing", ""),
            "job_levels": parse_job_levels(assessment.get("Job Levels", "")),
            "languages": parse_languages(assessment.get("Languages", "")),
            "url": assessment.get("URL", ""),
            "adaptive_support": assessment.get("Adaptive Support", ""),
            "test_type_keys": key_codes,
            "test_type_categories": key_descriptions,
            "description": assessment.get("Description", "")  # Store description in metadata
        }
        
        # Extract and process assessment length/duration
        assessment_length = assessment.get("Assessment Length", "")
        duration_minutes = extract_minutes(assessment_length)
        if duration_minutes is not None:
            metadata["duration_minutes"] = duration_minutes
        
        # Get description and PDF texts
        description = assessment.get("Description", "")
        cleaned_pdfs = assessment.get("Cleaned PDF Data", [])
        
        # Collect all cleaned PDF texts
        all_pdf_texts = []
        pdf_links = []
        
        for pdf in cleaned_pdfs:
            pdf_link = pdf.get("PDF Link", "")
            cleaned_text = pdf.get("Cleaned Text", "")
            
            if cleaned_text:
                all_pdf_texts.append(cleaned_text)
                pdf_links.append(pdf_link)
        
        # Add PDF links to metadata
        if pdf_links:
            metadata["pdf_links"] = pdf_links
        
        # Create a single document ID for the assessment
        doc_id = generate_document_id(assessment_name)
        
        # Prioritize PDF content, fall back to description if no PDFs
        if all_pdf_texts:
            # Combine all PDF texts into one document without splitting
            combined_text = f"{assessment_name}\n\n" + "\n\n".join(all_pdf_texts)
            metadata["content_type"] = "pdf_content"
            metadata["pdf_count"] = len(all_pdf_texts)
        elif description:
            # Use description as fallback
            combined_text = f"{assessment_name}\n\n{description}"
            metadata["content_type"] = "description_fallback"
        else:
            # Skip if no content available
            print(f"Warning: No content found for {assessment_name}, skipping")
            continue
            
        # Create a single document per assessment
        metadata["doc_id"] = doc_id
        documents.append({
            "id": doc_id,
            "text": combined_text,
            "metadata": metadata
        })
    
    print(f"Created {len(documents)} documents (one per assessment)")
    return documents

def store_embeddings_in_qdrant(documents, embeddings, qdrant_manager, batch_size=32):
    """Store embeddings and metadata in Qdrant with batch processing and improved progress tracking."""
    total_docs = len(documents)
    print(f"Processing {total_docs} documents...")
    
    # Calculate total number of batches
    total_batches = (total_docs - 1) // batch_size + 1
    
    # Track successful and failed operations
    successful_docs = 0
    failed_docs = 0
    
    # Process in batches with progress bar for batches
    for i in tqdm(range(0, total_docs, batch_size), desc="Processing batches", total=total_batches):
        batch = documents[i:i+batch_size]
        batch_size_actual = len(batch)
        
        try:
            # Extract texts and metadata
            texts = [doc["text"] for doc in batch]
            all_metadata = [doc["metadata"] for doc in batch]
            doc_ids = [doc["id"] for doc in batch]
            
            # Generate embeddings for the batch
            batch_embeddings = embeddings.embed_documents(texts)
            
            # Add embeddings to Qdrant in batch
            for j, (vector, metadata, doc_id) in enumerate(zip(batch_embeddings, all_metadata, doc_ids)):
                try:
                    # Add the embedding to Qdrant with metadata
                    qdrant_manager.add_embedding(
                        vector=vector,
                        payload=metadata,
                        id=doc_id  # Use our custom document ID
                    )
                    successful_docs += 1
                except Exception as e:
                    print(f"Error processing document {i+j}, ID {doc_id}: {e}")
                    failed_docs += 1
                
        except Exception as e:
            print(f"Error processing batch starting at document {i}: {e}")
            failed_docs += batch_size_actual
    
    print(f"Processing complete: {successful_docs} successful, {failed_docs} failed")
    print(f"Documents successfully stored in Qdrant collection '{qdrant_manager.collection_name}'")

if __name__ == "__main__":
    # Initialize embeddings
    embeddings = initialize_embeddings()

    # Load and process data
    data = load_data(INPUT_JSON)
    documents = extract_data_with_metadata(data)

    # Get the embedding dimension dynamically
    vector_size = get_embedding_dimension(embeddings)
    
    # Initialize QdrantManager
    qdrant_manager = QdrantManager(
        collection_name=QDRANT_COLLECTION_NAME,
        vector_size=vector_size
    )

    # Store embeddings in Qdrant with batching for GPU efficiency
    store_embeddings_in_qdrant(documents, embeddings, qdrant_manager, batch_size=32)
    
    print("Embeddings generation complete with optimized document handling.")