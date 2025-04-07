import os
import torch
from typing import Dict, List, Any, Optional
import json
import sys
from config import TEST_TYPE_MAPPING

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from embeddings.vector_db import QdrantManager
from embeddings.test import create_qdrant_filter
from embeddings.embeddings_client import TogetherEmbeddingsClient

class RetrieverAgent:
    """
    Agent that retrieves relevant assessments from the vector database
    using both semantic search and metadata filtering.
    """
    
    def __init__(self, collection_name: str = "shl_assessments_with_metadata"):
        """
        Initialize the Retriever agent with connections to the vector database.
        
        Args:
            collection_name: Name of the collection in Qdrant
        """
        self.collection_name = collection_name
        
        # Initialize Together AI client for embeddings
        self.embeddings = TogetherEmbeddingsClient()
        
        # Get the embedding dimension
        vector_size = 1024  # For BGE-large-en-v1.5
        
        # Initialize QdrantManager
        self.qdrant_manager = QdrantManager(
            collection_name=self.collection_name,
            vector_size=vector_size
        )
    
    def _prepare_filters(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the enhanced query metadata into Qdrant filters.
        
        Args:
            metadata: Dictionary with metadata from query enhancer
            
        Returns:
            Dictionary with filter conditions for Qdrant
        """
        filters = {}
        
        # Ensure job levels are a list
        if "job_levels" in metadata and metadata["job_levels"]:
            filters["job_levels"] = metadata["job_levels"] if isinstance(metadata["job_levels"], list) else [metadata["job_levels"]]
            
        # Ensure test types are a list
        if "test_types" in metadata and metadata["test_types"]:
            filters["test_type_categories"] = metadata["test_types"] if isinstance(metadata["test_types"], list) else [metadata["test_types"]]
            
        # Ensure languages are a list
        if "languages" in metadata and metadata["languages"]:
            filters["languages"] = metadata["languages"] if isinstance(metadata["languages"], list) else [metadata["languages"]]
        
        # Handle duration
        if "duration_minutes" in metadata and metadata["duration_minutes"]:
            filters["duration_minutes"] = {"max": metadata["duration_minutes"]}
        
        # Handle remote testing
        if "remote_testing" in metadata and metadata["remote_testing"] is not None:
            filters["remote_testing"] = "Yes" if metadata["remote_testing"] else "No"
        
        # Handle adaptive support
        if "adaptive_support" in metadata and metadata["adaptive_support"] is not None:
            filters["adaptive_support"] = "Yes" if metadata["adaptive_support"] else "No"
        
        return filters
    
    def retrieve_combined(self, query: str, metadata: Dict[str, Any] = None, k_per_method: int = 10) -> Dict[str, Any]:
        """
        Retrieve assessments using both filtered and unfiltered approaches and combine them.
        
        Args:
            query: Raw user query string
            metadata: Dictionary with extracted metadata from QueryEnhancer
            k_per_method: Number of results to retrieve for each method (filtered/unfiltered)
            
        Returns:
            Dictionary with both filtered and unfiltered results
        """
        # Get results with filters
        filtered_results = None
        if metadata:
            filtered_results = self.retrieve(query, metadata, first_stage_k=k_per_method, use_filters=True)
            print(f"Retrieved {len(filtered_results)} results with filters")
            
        # Get results without filters
        unfiltered_results = self.retrieve(query, metadata=None, first_stage_k=k_per_method, use_filters=False)
        print(f"Retrieved {len(unfiltered_results)} results without filters")
        
        # Combine results
        combined_results = (filtered_results or []) + (unfiltered_results or [])
        
        # Deduplicate combined results
        deduplicated_results = self._deduplicate_results(combined_results)
        print(f"Deduplicated to {len(deduplicated_results)} unique results")
        
        return {
            "query": query,
            "metadata": metadata,
            "filtered_results": filtered_results,
            "unfiltered_results": unfiltered_results,
            "deduplicated_results": deduplicated_results,  # Add deduplicated results
            "original_query": query
        }
    
    def retrieve(self, query: str, metadata: Dict[str, Any] = None, 
                 first_stage_k: int = 10, use_filters: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve top assessments matching the query and metadata filters.
        
        Args:
            query: Raw user query string
            metadata: Dictionary with extracted metadata from QueryEnhancer (optional)
            first_stage_k: Number of results to return for reranking
            use_filters: Whether to apply metadata filters
            
        Returns:
            List of assessment dictionaries with metadata and similarity scores
        """
        # Convert any 'null' values in metadata to None
        if metadata:
            metadata = {key: (None if value is None or value == "null" else value) for key, value in metadata.items()}
        
        # For BGE models, adding an instruction prefix improves search quality
        instruction = "Represent this sentence for searching relevant passages: "
        query_with_instruction = f"{instruction}{query}"
        
        # Generate embedding for the query
        query_embedding = self.embeddings.embed_query(query_with_instruction)
        
        # Prepare filter from metadata if specified
        search_filter = None
        if metadata and use_filters:
            filters = self._prepare_filters(metadata)
            
            if filters:
                # Log applied filters
                print(f"Applying filters: {json.dumps(filters, indent=2)}")
                
                # Create filter object for Qdrant if filters exist
                filter_conditions = create_qdrant_filter(filters)
                
                # Convert the filter object to a dictionary representation
                if filter_conditions:
                    search_filter = {}
                    if hasattr(filter_conditions, "must"):
                        search_filter["must"] = filter_conditions.must
                    if hasattr(filter_conditions, "should"):
                        search_filter["should"] = filter_conditions.should
        
        # Search Qdrant with the query embedding and filters
        results = self.qdrant_manager.search(
            query_vector=query_embedding,
            filter_conditions=search_filter,
            limit=first_stage_k
        )
        
        # Convert results to dictionaries with enhanced structure for reranking
        assessment_results = []
        for i, result in enumerate(results):
            # Extract essential data
            payload = dict(result.payload)
            assessment_name = payload.get("assessment_name", "Unknown Assessment")
            
            # Format text for reranking - combine name with description
            description = payload.get("description", "")
            # Convert test_type_keys to their long form using the mapping
            test_type_keys = payload.get('test_type_keys', [])
            test_types_long_form = []
            for key in test_type_keys:
                if key in TEST_TYPE_MAPPING:
                    test_types_long_form.append(TEST_TYPE_MAPPING[key])
                else:
                    test_types_long_form.append(key)  # Keep the original if not in mapping
            
            # Modified content_for_reranking in the retriever
            content_for_reranking = f"{assessment_name}\n\n"

            # Add description if available
            if description:
                content_for_reranking += f"{description}\n\n"
                
            # Include key extracted text from PDFs if available
            if "extracted_text" in payload and payload["extracted_text"]:
                # Extract a reasonable portion (first 1000 chars or so)
                content_for_reranking += f"{payload['extracted_text'][:1000]}\n\n"
                
            # Add key metadata as text with clear section headers
            content_for_reranking += "--- Assessment Metadata ---\n"
            content_for_reranking += f"Job Levels: {', '.join(payload.get('job_levels', ['Not specified']))}\n"
            content_for_reranking += f"Test Types: {', '.join(payload.get('test_type_categories', ['Not specified']))}\n"
                
            content_for_reranking += f"Languages: {', '.join(payload.get('languages', ['Not specified']))}\n"
            content_for_reranking += f"Duration: {payload.get('duration_minutes', 'Not specified')} minutes\n"
            content_for_reranking += f"Remote Testing: {payload.get('remote_testing', 'Not specified')}\n"
            content_for_reranking += f"Adaptive Support: {payload.get('adaptive_support', 'Not specified')}\n"
            
            if "skills_assessed" in payload:
                content_for_reranking += f"Skills Assessed: {', '.join(payload['skills_assessed'])}\n"
            elif "skills" in payload:
                content_for_reranking += f"Skills: {', '.join(payload.get('skills', ['Not specified']))}\n"
            
            # Include all fields from payload
            assessment_dict = {
                "id": payload.get("doc_id", f"result_{i}"),
                "rank": i + 1,                          # Initial rank based on embedding similarity
                "score": float(result.score),           # Embedding similarity score
                "name": assessment_name,                # Assessment name
                "content_for_reranking": content_for_reranking,  # Text for reranker
                "metadata": payload,                    # All original metadata
                "source": "filtered" if use_filters else "unfiltered"  # Track source for debugging
            }
            
            assessment_results.append(assessment_dict)
            
        return assessment_results
    
    def get_collection_stats(self):
        """Get statistics about the vector database collection."""
        try:
            collection_info = self.qdrant_manager.qdrant_client.get_collection(self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": str(e)}
        
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate results based on a unique identifier (e.g., doc_id or name).
        
        Args:
            results: List of assessment dictionaries
            
        Returns:
            Deduplicated list of assessment dictionaries
        """
        seen_ids = set()
        deduplicated_results = []
        
        for result in results:
            # Use `doc_id` if available, otherwise fall back to `name`
            unique_id = result["metadata"].get("doc_id", result["name"])
            if unique_id not in seen_ids:
                seen_ids.add(unique_id)
                deduplicated_results.append(result)
        
        return deduplicated_results

# Test the agent if run directly
if __name__ == "__main__":
    # Instantiate the agent
    retriever = RetrieverAgent()
    
    # Print collection stats
    print("Vector DB Stats:", retriever.get_collection_stats())
    
    # Example query and metadata
    # query = "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
    # metadata = {
    #     "test_types": [
    #         "Knowledge and Skills",
    #         "Ability and Aptitude"
    #     ],
    #     "skills": [
    #         "Python",
    #         "SQL",
    #         "JavaScript"
    #     ],
    #     "adaptive_support": None,
    #     "duration_minutes": 60,
    #     "job_levels": [
    #         "Mid-Professional"
    #     ],
    #     "languages": [
    #         "English (USA)"
    #     ],
    #     "remote_testing": None
    # }
    query = "Looking for retail store manager assessment that evaluates inventory management skills"
    metadata = {
        "test_types": ["Knowledge and Skills", "Ability and Aptitude"],
        "skills": ["Inventory Management", "Customer Service", "Team Management"],
        "adaptive_support": None,
        "duration_minutes": 45,
        "job_levels": ["Front Line Manager"],
        "languages": ["English (USA)"],
        "remote_testing": None
    }

    # Test combined retrieval
    results = retriever.retrieve_combined(query, metadata, k_per_method=10)
    
    # Print top 5 from filtered results
    if results["filtered_results"]:
        print("\n--- Top 5 filtered results ---")
        for i, sample in enumerate(results["filtered_results"][:5]):
            print(f"Result {i + 1}:")
            for key, value in sample.items():
                print(f"  {key}: {value}")
            print()
    
    # Print top 5 from unfiltered results
    if results["unfiltered_results"]:
        print("\n--- Top 5 unfiltered results ---")
        for i, sample in enumerate(results["unfiltered_results"][:5]):
            print(f"Result {i + 1}:")
            for key, value in sample.items():
                print(f"  {key}: {value}")
            print()

# import os
# import torch
# from typing import Dict, List, Any, Optional
# import json
# import sys
# from config import TEST_TYPE_MAPPING
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# # Ensure the project root is in the path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from embeddings.vector_db import QdrantManager

# # Load environment variables to get API key
# load_dotenv()

# class RetrieverAgent:
#     """
#     Agent that retrieves relevant assessments from the vector database
#     using both semantic search and metadata filtering.
#     """
    
#     def __init__(self, collection_name: str = "shl_assessments_with_metadata"):
#         """
#         Initialize the Retriever agent with connections to the vector database.
        
#         Args:
#             collection_name: Name of the collection in Qdrant
#         """
#         self.collection_name = collection_name
        
#         # Initialize Hugging Face Inference Client
#         hf_api_key = os.getenv("HF_API_KEY")
#         if not hf_api_key:
#             raise ValueError("HF_API_KEY not found in environment. Please set it in your .env file.")
        
#         self.client = InferenceClient(
#             model="BAAI/bge-large-en-v1.5",
#             token=hf_api_key
#         )
        
#         # Initialize QdrantManager
#         self.qdrant_manager = QdrantManager(
#             collection_name=self.collection_name,
#             vector_size=1024  # BGE-large-en-v1.5 has 1024-dimensional embeddings
#         )
        
#         print(f"RetrieverAgent initialized with Hugging Face Inference API for embeddings")
    
#     def embed_query(self, query_text: str) -> List[float]:
#         """
#         Generate embedding for a query text using Hugging Face Inference API.
        
#         Args:
#             query_text: The query text to embed
            
#         Returns:
#             Embedding vector as a list of floats
#         """
#         # Add the instruction prefix for retrieval
#         query_with_instruction = f"Represent this sentence for searching relevant passages: {query_text}"
        
#         try:
#             # Call the feature-extraction endpoint
#             embedding = self.client.feature_extraction(
#                 text=query_with_instruction
#             )
            
#             # Convert numpy array to list if needed
#             if hasattr(embedding, 'tolist'):  # Check if it's a numpy array
#                 embedding = embedding.tolist()
#             elif isinstance(embedding, str):
#                 # In case it returns a string representation of a list
#                 embedding = json.loads(embedding)
            
#             if isinstance(embedding, list):
#                 # If it's a nested list (batch processing), take the first item
#                 if embedding and isinstance(embedding[0], list):
#                     embedding = embedding[0]
                    
#                 # Normalize the embedding
#                 norm = sum(x * x for x in embedding) ** 0.5
#                 if norm > 0:
#                     normalized_embedding = [x / norm for x in embedding]
#                     return normalized_embedding
#                 return embedding
#             else:
#                 print(f"Warning: Unexpected embedding format type: {type(embedding)}")
#                 # Try to convert to list as a last resort
#                 try:
#                     embedding_list = list(embedding)
#                     # Normalize
#                     norm = sum(x * x for x in embedding_list) ** 0.5
#                     if norm > 0:
#                         return [x / norm for x in embedding_list]
#                     return embedding_list
#                 except:
#                     raise ValueError(f"Cannot convert embedding to list: {embedding}")
                
#         except Exception as e:
#             print(f"Error generating embedding: {e}")
#             raise
    
#     def _prepare_filters(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
#         """Create filter conditions from query metadata."""
#         filters = {}
        
#         # Apply job level filter
#         if "job_levels" in metadata and metadata["job_levels"]:
#             filters["job_levels"] = metadata["job_levels"]
            
#         # Apply test type filter
#         if "test_types" in metadata and metadata["test_types"]:
#             filters["test_type_categories"] = metadata["test_types"]
            
#         # Apply language filter
#         if "languages" in metadata and metadata["languages"]:
#             filters["languages"] = metadata["languages"]
            
#         # Apply duration filter
#         if "duration_minutes" in metadata and metadata["duration_minutes"]:
#             filters["duration_minutes"] = {"max": metadata["duration_minutes"]}
        
#         # Handle remote testing
#         if "remote_testing" in metadata and metadata["remote_testing"] is not None:
#             filters["remote_testing"] = "Yes" if metadata["remote_testing"] else "No"
        
#         # Handle adaptive support
#         if "adaptive_support" in metadata and metadata["adaptive_support"] is not None:
#             filters["adaptive_support"] = "Yes" if metadata["adaptive_support"] else "No"
        
#         return filters
        
#     def retrieve(self, query: str, metadata: Dict[str, Any] = None, 
#                  first_stage_k: int = 10, use_filters: bool = True) -> List[Dict[str, Any]]:
#         """
#         Retrieve top assessments matching the query and metadata filters.
        
#         Args:
#             query: User query
#             metadata: Dictionary with query metadata for filtering
#             first_stage_k: Number of top results to retrieve
#             use_filters: Whether to apply metadata filters
            
#         Returns:
#             List of assessment dictionaries with scores
#         """
#         # Get query embedding using the HF API
#         query_embedding = self.embed_query(query)
        
#         # Prepare filter from metadata if specified
#         search_filter = None
#         if metadata and use_filters:
#             filters = self._prepare_filters(metadata)
            
#             if filters:
#                 # Log applied filters
#                 print(f"Applying filters: {json.dumps(filters, indent=2)}")
                
#                 # Create filter object for Qdrant
#                 filter_conditions = create_qdrant_filter(filters)
                
#                 # Convert to dictionary representation for Qdrant
#                 if filter_conditions:
#                     search_filter = {}
#                     if hasattr(filter_conditions, "must"):
#                         search_filter["must"] = filter_conditions.must
#                     if hasattr(filter_conditions, "should"):
#                         search_filter["should"] = filter_conditions.should
        
#         # Search Qdrant with the query embedding and filters
#         results = self.qdrant_manager.search(
#             query_vector=query_embedding,
#             filter_conditions=search_filter,
#             limit=first_stage_k
#         )
        
#         # Convert results to dictionaries with enhanced structure for reranking
#         assessment_results = []
#         for i, result in enumerate(results):
#             # Extract essential data
#             payload = dict(result.payload)
#             assessment_name = payload.get("assessment_name", "Unknown Assessment")
            
#             # Format text for reranking - combine name with description
#             description = payload.get("description", "")
#             # Convert test_type_keys to their long form using the mapping
#             test_type_keys = payload.get('test_type_keys', [])
#             test_types_long_form = []
#             for key in test_type_keys:
#                 if key in TEST_TYPE_MAPPING:
#                     test_types_long_form.append(TEST_TYPE_MAPPING[key])
#                 else:
#                     test_types_long_form.append(key)  # Keep the original if not in mapping
            
#             # Modified content_for_reranking in the retriever
#             content_for_reranking = f"{assessment_name}\n\n"

#             # Add description if available
#             if description:
#                 content_for_reranking += f"{description}\n\n"
                
#             # Include key extracted text from PDFs if available
#             if "extracted_text" in payload and payload["extracted_text"]:
#                 # Extract a reasonable portion (first 1000 chars or so)
#                 content_for_reranking += f"{payload['extracted_text'][:1000]}\n\n"
                
#             # Add key metadata as text with clear section headers
#             content_for_reranking += "--- Assessment Metadata ---\n"
#             content_for_reranking += f"Job Levels: {', '.join(payload.get('job_levels', ['Not specified']))}\n"
#             content_for_reranking += f"Test Types: {', '.join(payload.get('test_type_categories', ['Not specified']))}\n"
                
#             content_for_reranking += f"Languages: {', '.join(payload.get('languages', ['Not specified']))}\n"
#             content_for_reranking += f"Duration: {payload.get('duration_minutes', 'Not specified')} minutes\n"
#             content_for_reranking += f"Remote Testing: {payload.get('remote_testing', 'Not specified')}\n"
#             content_for_reranking += f"Adaptive Support: {payload.get('adaptive_support', 'Not specified')}\n"
            
#             if "skills_assessed" in payload:
#                 content_for_reranking += f"Skills Assessed: {', '.join(payload['skills_assessed'])}\n"
#             elif "skills" in payload:
#                 content_for_reranking += f"Skills: {', '.join(payload.get('skills', ['Not specified']))}\n"
            
#             # Include all fields from payload
#             assessment_dict = {
#                 "id": payload.get("doc_id", f"result_{i}"),
#                 "rank": i + 1,                          # Initial rank based on embedding similarity
#                 "score": float(result.score),           # Embedding similarity score
#                 "name": assessment_name,                # Assessment name
#                 "content_for_reranking": content_for_reranking,  # Text for reranker
#                 "metadata": payload,                    # All original metadata
#                 "source": "filtered" if use_filters else "unfiltered"  # Track source for debugging
#             }
            
#             assessment_results.append(assessment_dict)
            
#         return assessment_results
    
#     def get_collection_stats(self):
#         """Get statistics about the vector database collection."""
#         try:
#             collection_info = self.qdrant_manager.qdrant_client.get_collection(self.collection_name)
#             return {
#                 "vectors_count": collection_info.vectors_count,
#                 "indexed_vectors_count": collection_info.indexed_vectors_count,
#                 "status": collection_info.status
#             }
#         except Exception as e:
#             return {"error": str(e)}
        
#     def retrieve_combined(self, query: str, metadata: Dict[str, Any] = None, k_per_method: int = 10) -> Dict[str, Any]:
#         """
#         Retrieve assessments using both filtered and unfiltered approaches and combine them.
        
#         Args:
#             query: Raw user query string
#             metadata: Dictionary with extracted metadata from QueryEnhancer
#             k_per_method: Number of results to retrieve for each method
            
#         Returns:
#             Dictionary with both filtered and unfiltered results
#         """
#         # Get results with filters
#         filtered_results = []
#         if metadata:
#             try:
#                 filtered_results = self.retrieve(query, metadata, first_stage_k=k_per_method, use_filters=True)
#                 print(f"Retrieved {len(filtered_results)} results with filters")
#             except Exception as e:
#                 print(f"Error retrieving filtered results: {e}")
        
#         # Get results without filters
#         unfiltered_results = []
#         try:
#             unfiltered_results = self.retrieve(query, metadata=None, first_stage_k=k_per_method, use_filters=False)
#             print(f"Retrieved {len(unfiltered_results)} results without filters")
#         except Exception as e:
#             print(f"Error retrieving unfiltered results: {e}")
        
#         # Return combined results
#         return {
#             "query": query,
#             "metadata": metadata,
#             "filtered_results": filtered_results,
#             "unfiltered_results": unfiltered_results,
#             "original_query": query
#         }
        
#     # Add this function (copied from embeddings/test.py)
# from qdrant_client.http import models

# def create_qdrant_filter(filters: Dict[str, Any]) -> Optional[models.Filter]:
#     """Create a Qdrant filter from a dictionary of filter conditions."""
#     if not filters:
#         return None
    
#     must_conditions = []
    
#     # Process job levels filter
#     if "job_levels" in filters:
#         job_levels = filters["job_levels"]
#         if isinstance(job_levels, list) and job_levels:
#             job_level_conditions = []
#             for level in job_levels:
#                 job_level_conditions.append(
#                     models.FieldCondition(
#                         key="job_levels",
#                         match=models.MatchValue(value=level)
#                     )
#                 )
#             must_conditions.append(
#                 models.Filter(
#                     should=job_level_conditions
#                 )
#             )
    
#     # Process test type categories filter
#     if "test_type_categories" in filters:
#         test_types = filters["test_type_categories"]
#         if isinstance(test_types, list) and test_types:
#             test_type_conditions = []
#             for test_type in test_types:
#                 test_type_conditions.append(
#                     models.FieldCondition(
#                         key="test_type_categories",
#                         match=models.MatchValue(value=test_type)
#                     )
#                 )
#             must_conditions.append(
#                 models.Filter(
#                     should=test_type_conditions
#                 )
#             )
    
#     # Process languages filter
#     if "languages" in filters:
#         languages = filters["languages"]
#         if isinstance(languages, list) and languages:
#             language_conditions = []
#             for language in languages:
#                 language_conditions.append(
#                     models.FieldCondition(
#                         key="languages",
#                         match=models.MatchValue(value=language)
#                     )
#                 )
#             must_conditions.append(
#                 models.Filter(
#                     should=language_conditions
#                 )
#             )
    
#     # Process duration filter
#     if "duration_minutes" in filters and isinstance(filters["duration_minutes"], dict):
#         duration_filter = filters["duration_minutes"]
#         if "max" in duration_filter:
#             must_conditions.append(
#                 models.FieldCondition(
#                     key="duration_minutes",
#                     range=models.Range(
#                         lte=duration_filter["max"]
#                     )
#                 )
#             )
    
#     # Process remote testing
#     if "remote_testing" in filters and filters["remote_testing"]:
#         must_conditions.append(
#             models.FieldCondition(
#                 key="remote_testing",
#                 match=models.MatchValue(value=filters["remote_testing"])
#             )
#         )
    
#     # Process adaptive support
#     if "adaptive_support" in filters and filters["adaptive_support"]:
#         must_conditions.append(
#             models.FieldCondition(
#                 key="adaptive_support",
#                 match=models.MatchValue(value=filters["adaptive_support"])
#             )
#         )
    
#     if must_conditions:
#         return models.Filter(
#             must=must_conditions
#         )
    
#     return None

# # Test the agent if run directly
# if __name__ == "__main__":
#     # Instantiate the agent
#     retriever = RetrieverAgent()
    
#     # Print collection stats
#     print("Vector DB Stats:", retriever.get_collection_stats())
    
#     # Example query and metadata
#     # query = "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
#     # metadata = {
#     #     "test_types": [
#     #         "Knowledge and Skills",
#     #         "Ability and Aptitude"
#     #     ],
#     #     "skills": [
#     #         "Python",
#     #         "SQL",
#     #         "JavaScript"
#     #     ],
#     #     "adaptive_support": None,
#     #     "duration_minutes": 60,
#     #     "job_levels": [
#     #         "Mid-Professional"
#     #     ],
#     #     "languages": [
#     #         "English (USA)"
#     #     ],
#     #     "remote_testing": None
#     # }
#     query = "Looking for retail store manager assessment that evaluates inventory management skills"
#     metadata = {
#         "test_types": ["Knowledge and Skills", "Ability and Aptitude"],
#         "skills": ["Inventory Management", "Customer Service", "Team Management"],
#         "adaptive_support": None,
#         "duration_minutes": 45,
#         "job_levels": ["Front Line Manager"],
#         "languages": ["English (USA)"],
#         "remote_testing": None
#     }

#     # Test combined retrieval
#     results = retriever.retrieve_combined(query, metadata, k_per_method=10)
    
#     # Print top 5 from filtered results
#     if results["filtered_results"]:
#         print("\n--- Top 5 filtered results ---")
#         for i, sample in enumerate(results["filtered_results"][:5]):
#             print(f"Result {i + 1}:")
#             for key, value in sample.items():
#                 print(f"  {key}: {value}")
#             print()
    
#     # Print top 5 from unfiltered results
#     if results["unfiltered_results"]:
#         print("\n--- Top 5 unfiltered results ---")
#         for i, sample in enumerate(results["unfiltered_results"][:5]):
#             print(f"Result {i + 1}:")
#             for key, value in sample.items():
#                 print(f"  {key}: {value}")
#             print()

