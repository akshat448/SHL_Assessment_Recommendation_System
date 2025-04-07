import os
import torch
from typing import Dict, List, Any, Optional
import json
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RerankerAgent:
    """
    Agent that reranks candidate assessments using the BGE reranker model.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model_cache_dir: str = None, use_fp16: bool = True):
        """
        Initialize the Reranker agent with the BGE reranker model.
        
        Args:
            model_name: Name of the reranker model
            model_cache_dir: Directory to cache the model (optional)
            use_fp16: Whether to use half precision for faster inference
        """
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir or "/Users/akshat/Developer/Tasks/SHL/.model_cache"
        self.use_fp16 = use_fp16
        
        print(f"Loading reranker model: {model_name}")
        
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=self.model_cache_dir)
        
        # # Enable FP16 if requested and supported
        # if self.use_fp16 and torch.backends.mps.is_available():
        #     self.model = self.model.half()
            
        # # Move model to appropriate device
        # if torch.cuda.is_available():
        #     device = "cuda"
        #     self.model = self.model.to(device)
        #     print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device = "mps"
        #     self.model = self.model.to(device)
        #     print("Using MPS (Metal Performance Shaders) on macOS.")
        # else:
        device = "cpu"
        print("GPU acceleration not available, using CPU. This will be slower.")
    
        self.device = torch.device(device)
        self.model.eval()  # Set to evaluation mode
        
    def rerank(self, query: str, retriever_results: Dict[str, Any], top_k: int = 10, 
           retriever_weight: float = 0.4, reranker_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        Rerank the retrieved assessment candidates and return the top-k unique results.
        
        Args:
            query: Original query string
            retriever_results: Results from RetrieverAgent's retrieve_combined method
            top_k: Number of top unique results to return after reranking
            retriever_weight: Weight to give original retriever score (0.0-1.0)
            reranker_weight: Weight to give reranker score (0.0-1.0)
            
        Returns:
            List of reranked assessment dictionaries without duplicates
        """
        # Combine filtered and unfiltered results with deduplication
        unique_candidates = {}  # Use a dictionary to track unique assessments
        
        # Process filtered results first (they're more likely to be relevant)
        if "filtered_results" in retriever_results and retriever_results["filtered_results"]:
            for candidate in retriever_results["filtered_results"]:
                # Use name as the unique identifier (or doc_id if available)
                unique_id = candidate["metadata"].get("doc_id", candidate["name"])
                unique_candidates[unique_id] = candidate
            
        # Then process unfiltered results, only adding if not already present or if better score
        if "unfiltered_results" in retriever_results and retriever_results["unfiltered_results"]:
            for candidate in retriever_results["unfiltered_results"]:
                unique_id = candidate["metadata"].get("doc_id", candidate["name"])
                # Only add if not already present or if this has a better score
                if unique_id not in unique_candidates or candidate["score"] > unique_candidates[unique_id]["score"]:
                    unique_candidates[unique_id] = candidate
        
        # Convert back to list
        all_candidates = list(unique_candidates.values())
            
        if not all_candidates:
            print("No candidates to rerank!")
            return []
        
        print(f"Reranking {len(all_candidates)} unique candidates...")
        
        # Prepare pairs for reranking
        pairs = []
        for candidate in all_candidates:
            # Use content_for_reranking which combines name and description
            pairs.append([query, candidate["content_for_reranking"]])
            
        # Process in batches to avoid memory issues
        batch_size = 8  # Adjust based on available memory
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Tokenize the batch
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True,
                    return_tensors='pt', 
                    max_length=512
                ).to(self.device)
                
                # Get reranker scores
                outputs = self.model(**inputs)
                scores = outputs.logits.view(-1).cpu().float().tolist()
                all_scores.extend(scores)
        
        # Find min/max scores for normalization
        if all_scores:
            min_rerank_score = min(all_scores)
            max_rerank_score = max(all_scores)
            score_range = max_rerank_score - min_rerank_score
        else:
            min_rerank_score = 0
            score_range = 1
        
        # Get original score range for normalization
        original_scores = [candidate["score"] for candidate in all_candidates]
        if original_scores:
            min_original_score = min(original_scores)
            max_original_score = max(original_scores)
            original_range = max_original_score - min_original_score
        else:
            min_original_score = 0
            original_range = 1
                
        # Add reranker scores and calculate combined scores for candidates
        for i, score in enumerate(all_scores):
            # Store the raw reranker score
            all_candidates[i]["rerank_score"] = score
            
            # Normalize both scores to 0-1 range to make them comparable
            # if score_range > 0:
            #     normalized_rerank = (score - min_rerank_score) / score_range
            #     #normalized_rerank = 1 / (1 + torch.exp(torch.tensor(-score)))  # Sigmoid
            # else:
            #     normalized_rerank = 0.5  # Default if all scores are the same
            temperature = 2.0  # You can tune this
            normalized_rerank = 1 / (1 + math.exp(-score / temperature))

            if original_range > 0:
                normalized_retriever = (all_candidates[i]["score"] - min_original_score) / original_range
            else:
                normalized_retriever = 0.5  # Default if all scores are the same
            
            # Calculate combined score using the weights
            combined_score = (retriever_weight * normalized_retriever) + (reranker_weight * normalized_rerank)
            all_candidates[i]["combined_score"] = combined_score
                
        # Sort by the combined score (descending)
        reranked_candidates = sorted(all_candidates, key=lambda x: x["combined_score"], reverse=True)
        
        # Return top-k results
        top_results = reranked_candidates[:top_k]
        
        # Add reranking info to each result
        for i, result in enumerate(top_results):
            result["rerank_position"] = i + 1
            result["original_rank"] = result["rank"]
            result["rank"] = i + 1  # Update rank to reranked position
            
        return top_results
        
    def compute_single_score(self, query: str, text: str) -> float:
        """Calculate reranker score for a single query-text pair."""
        with torch.no_grad():
            inputs = self.tokenizer(
                [[query, text]], 
                padding=True, 
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            score = outputs.logits.view(-1).cpu().float().item()
            
        return score
        
# Test the agent if run directly
if __name__ == "__main__":
    from retriever import RetrieverAgent
    
    # Instantiate the agents
    retriever = RetrieverAgent()
    reranker = RerankerAgent()
    
    # Example query and metadata
    query = "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
    metadata = {
        "test_types": [
            "Knowledge and Skills",
            "Ability and Aptitude"
        ],
        "skills": [
            "Python",
            "SQL",
            "JavaScript"
        ],
        "adaptive_support": None,
        "duration_minutes": 60,
        "job_levels": [
            "Mid-Professional"
        ],
        "languages": [
            "English (USA)"
        ],
        "remote_testing": None
    }
    # query = "Looking for retail store manager assessment that evaluates inventory management skills"
    # metadata = {
    #     "test_types": ["Knowledge and Skills", "Ability and Aptitude"],
    #     "skills": ["Inventory Management", "Customer Service", "Team Management"],
    #     "adaptive_support": None,
    #     "duration_minutes": 45,
    #     "job_levels": ["Front Line Manager"],
    #     "languages": ["English (USA)"],
    #     "remote_testing": None
    # }
    # First retrieve candidates
    retriever_results = retriever.retrieve_combined(query, metadata, k_per_method=25)
    
    # Then rerank them - with custom weights (0.4 for retriever, 0.6 for reranker)
    reranked_results = reranker.rerank(query, retriever_results, top_k=10, 
                                        retriever_weight=0.7, reranker_weight=0.3)
    
    # Print the reranked results
    print("\n--- Top Reranked Results ---")
    for i, result in enumerate(reranked_results[:10]):
        print(f"{i+1}. {result['name']}")
        print(f"   Original Rank: {result['original_rank']}")
        print(f"   Original Score: {result['score']:.4f}")
        print(f"   Rerank Score: {result['rerank_score']:.4f}")
        print(f"   Combined Score: {result.get('combined_score', 0):.4f}")  # Show combined score
        print(f"   Source: {result['source']}")
        print()

# import os
# import time
# import random
# from typing import Dict, List, Any, Optional
# import json
# import math
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# # Load environment variables
# load_dotenv()

# class RerankerAgent:
#     """
#     Agent that reranks candidate assessments using the BGE reranker model via Hugging Face Inference API.
#     """
    
#     def __init__(self, max_retries=3, batch_size=5):
#         """
#         Initialize the Reranker agent with the Hugging Face Inference API.
        
#         Args:
#             max_retries: Maximum number of retry attempts for API calls
#             batch_size: Number of items to process in a single batch
#         """
#         # Get API key from environment
#         hf_api_key = os.getenv("HF_API_KEY")
#         if not hf_api_key:
#             raise ValueError("HF_API_KEY not found in environment. Please set it in your .env file.")
        
#         # Initialize inference client - don't specify a model here
#         self.client = InferenceClient(token=hf_api_key)
        
#         self.max_retries = max_retries
#         self.batch_size = batch_size
#         self.model_id = "BAAI/bge-reranker-v2-m3"
        
#         print("RerankerAgent initialized with Hugging Face Inference API")
    
#     def compute_scores_batch(self, query: str, texts: List[str]) -> List[float]:
#         """
#         Calculate reranker scores for a batch of texts using Hugging Face Inference API.
        
#         Args:
#             query: Query string
#             texts: List of texts to score against the query
            
#         Returns:
#             List of scores, one for each text
#         """
#         if not texts:
#             return []
            
#         # Format inputs for the BGE reranker in the correct format
#         # BGE reranker expects inputs in the format [[query, document], [query, document], ...]
#         formatted_inputs = [[query, text] for text in texts]
        
#         # Implement retry logic with exponential backoff
#         retries = 0
#         max_wait = 16  # Maximum wait time in seconds
        
#         while retries <= self.max_retries:
#             try:
#                 # Use the raw inference endpoint with the correct payload format
#                 # This is the format that the model expects according to the documentation
#                 response = self.client.post(
#                     json={"inputs": formatted_inputs},
#                     model="BAAI/bge-reranker-v2-m3"
#                 )
                
#                 # Handle the response which should be a list of scores
#                 if isinstance(response, list):
#                     return [float(score) for score in response]
#                 else:
#                     print(f"Warning: Unexpected response format: {response}")
#                     return [0.5] * len(texts)
                    
#             except Exception as e:
#                 retries += 1
#                 if retries > self.max_retries:
#                     print(f"Error computing reranker scores after {self.max_retries} attempts: {e}")
#                     # Return default scores (0.5) if all retries fail
#                     return [0.5] * len(texts)
                
#                 # Exponential backoff with jitter
#                 wait_time = min(2 ** retries + random.uniform(0, 1), max_wait)
#                 print(f"API call failed, retrying in {wait_time:.2f}s (attempt {retries}/{self.max_retries})...")
#                 time.sleep(wait_time)
    
#     def rerank(self, query: str, retriever_results: Dict[str, Any], top_k: int = 10, 
#            retriever_weight: float = 0.4, reranker_weight: float = 0.6) -> List[Dict[str, Any]]:
#         """
#         Rerank the retrieved assessment candidates and return the top-k unique results.
        
#         Args:
#             query: Original query string
#             retriever_results: Results from RetrieverAgent's retrieve_combined method
#             top_k: Number of top unique results to return after reranking
#             retriever_weight: Weight to give original retriever score (0.0-1.0)
#             reranker_weight: Weight to give reranker score (0.0-1.0)
            
#         Returns:
#             List of reranked assessment dictionaries without duplicates
#         """
#         # Combine filtered and unfiltered results with deduplication
#         unique_candidates = {}  # Use a dictionary to track unique assessments
        
#         # Process filtered results first (they're more likely to be relevant)
#         if "filtered_results" in retriever_results and retriever_results["filtered_results"]:
#             for candidate in retriever_results["filtered_results"]:
#                 # Use name or ID as unique identifier
#                 unique_id = candidate.get("id", candidate.get("name", f"item_{len(unique_candidates)}"))
#                 unique_candidates[unique_id] = candidate
        
#         # Then process unfiltered results, only adding if not already present or if better score
#         if "unfiltered_results" in retriever_results and retriever_results["unfiltered_results"]:
#             for candidate in retriever_results["unfiltered_results"]:
#                 unique_id = candidate.get("id", candidate.get("name", f"item_{len(unique_candidates)}"))
#                 # Only add if not already present or if this has a better score
#                 if unique_id not in unique_candidates or candidate.get("score", 0) > unique_candidates[unique_id].get("score", 0):
#                     unique_candidates[unique_id] = candidate
        
#         # Convert back to list
#         all_candidates = list(unique_candidates.values())
            
#         if not all_candidates:
#             print("No candidates to rerank!")
#             return []
        
#         print(f"Reranking {len(all_candidates)} unique candidates...")
        
#         # Process candidates in batches to avoid API rate limits
#         all_rerank_scores = []
        
#         # Extract content for reranking from all candidates
#         contents = [candidate.get("content_for_reranking", candidate.get("name", "")) for candidate in all_candidates]
        
#         # Process in batches
#         for i in range(0, len(contents), self.batch_size):
#             batch = contents[i:i + self.batch_size]
#             batch_scores = self.compute_scores_batch(query, batch)
#             all_rerank_scores.extend(batch_scores)
            
#             # Add a small delay between batches to avoid rate limiting
#             if i + self.batch_size < len(contents):
#                 time.sleep(0.5)
        
#         # Find min/max scores for normalization
#         if all_rerank_scores:
#             min_rerank_score = min(all_rerank_scores)
#             max_rerank_score = max(all_rerank_scores)
#             score_range = max_rerank_score - min_rerank_score
#         else:
#             min_rerank_score = 0
#             score_range = 1
        
#         # Get original score range for normalization
#         original_scores = [candidate.get("score", 0) for candidate in all_candidates]
#         if original_scores:
#             min_original_score = min(original_scores)
#             max_original_score = max(original_scores)
#             original_range = max_original_score - min_original_score
#         else:
#             min_original_score = 0
#             original_range = 1
                
#         # Add reranker scores and calculate combined scores for candidates
#         for i, score in enumerate(all_rerank_scores):
#             # Store the raw reranker score
#             all_candidates[i]["rerank_score"] = score
            
#             # Use sigmoid normalization for reranker score
#             temperature = 2.0
#             normalized_rerank = 1 / (1 + math.exp(-score / temperature))
            
#             # Linear normalization for retriever score
#             if original_range > 0:
#                 normalized_retriever = (all_candidates[i].get("score", 0) - min_original_score) / original_range
#             else:
#                 normalized_retriever = 0.5  # Default if all scores are the same
                
#             # If reranker score is from a fallback (API failure), rely more on retriever
#             if score == 0.5 and all_rerank_scores.count(0.5) > 5:  # If many default scores, likely API failure
#                 retriever_weight = 0.8
#                 reranker_weight = 0.2

#             # Calculate combined score using the weights
#             combined_score = (retriever_weight * normalized_retriever) + (reranker_weight * normalized_rerank)
#             all_candidates[i]["combined_score"] = combined_score
                
#         # Sort by the combined score (descending)
#         reranked_candidates = sorted(all_candidates, key=lambda x: x.get("combined_score", 0), reverse=True)
        
#         # Return top-k results
#         top_results = reranked_candidates[:top_k]
        
#         # Add reranking info to each result
#         for i, result in enumerate(top_results):
#             result["rerank_position"] = i + 1
#             result["original_rank"] = result.get("rank", 0)
#             result["rank"] = i + 1  # Update rank to reranked position
            
#         return top_results

# if __name__ == "__main__":
#     # Example usage for testing
#     from retriever import RetrieverAgent
    
#     retriever = RetrieverAgent()
#     reranker = RerankerAgent(max_retries=2, batch_size=3)
    
#     # Simple test
#     query = "Java developer with team collaboration skills"
#     metadata = {
#         "test_types": ["Knowledge and Skills", "Ability and Aptitude"],
#         "skills": ["Java", "Team Collaboration"],
#         "duration_minutes": 40,
#         "job_levels": ["Mid-Professional"],
#         "languages": ["English (USA)"],
#     }
    
#     # Test
#     try:
#         retriever_results = retriever.retrieve_combined(query, metadata, k_per_method=5)
#         reranked_results = reranker.rerank(query, retriever_results, top_k=3)
        
#         print(f"\nTop results after reranking:")
#         for i, result in enumerate(reranked_results):
#             print(f"{i+1}. {result.get('name')} (Score: {result.get('combined_score', 0):.3f})")
#     except Exception as e:
#         print(f"Test failed: {e}")