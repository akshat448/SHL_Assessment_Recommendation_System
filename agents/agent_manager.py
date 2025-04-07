from typing import Dict, Any, Union
import json
import time

from query_enhancer import QueryEnhancerAgent
from retriever import RetrieverAgent
from reranker import RerankerAgent
from output_formatter import OutputFormatterAgent

class AgentManager:
    """
    Manager class that coordinates the multi-agent workflow.
    """
    
    def __init__(self, use_reranker: bool = True, use_llm_explanations: bool = True):
        """Initialize all agents."""
        print("Initializing agents...")
        self.query_enhancer = QueryEnhancerAgent()
        self.retriever = RetrieverAgent()
        
        if use_reranker:
            try:
                self.reranker = RerankerAgent()
                self.use_reranker = True
            except Exception as e:
                print(f"Error initializing reranker: {e}")
                self.use_reranker = False
        else:
            self.use_reranker = False
            
        self.formatter = OutputFormatterAgent(use_llm=use_llm_explanations)
    
    def process_query(self, query: str, metadata: Dict = None, top_k: int = 10, top_n_from_k: int = 10, 
                      format_as_text: bool = False) -> Union[Dict[str, Any], str]:
        """
        Process a user query through all agents in the pipeline.
        
        Args:
            query: User's natural language query
            metadata: Optional pre-enhanced metadata (to skip query enhancement)
            top_k: Number of results to retrieve from the retriever
            top_n_from_k: Number of top results to select from the reranker
            format_as_text: Whether to return results as text or JSON
            
        Returns:
            A JSON structure with the results
        """
        start_time = time.time()
        print(f"Processing query: '{query}'")
        
        # Step 1: Enhance the query with metadata (if not provided)
        if metadata is None:
            print("Enhancing query...")
            metadata = self.query_enhancer.enhance_query(query)
            print(f"Enhanced with metadata: {json.dumps(metadata, indent=2)}")
        else:
            print("Using pre-enhanced metadata")
        
        # Step 2: Retrieve relevant assessments
        print("Retrieving assessments...")
        retriever_results = self.retriever.retrieve_combined(query, metadata, k_per_method=top_k)
        print(f"Retriever results: {json.dumps(retriever_results, indent=2)}")
        
        filtered_count = len(retriever_results.get("filtered_results", [])) if retriever_results.get("filtered_results") else 0
        unfiltered_count = len(retriever_results.get("unfiltered_results", [])) if retriever_results.get("unfiltered_results") else 0
        deduplicated_count = len(retriever_results.get("deduplicated_results", []))
        print(f"Retrieved {filtered_count} results with filters, {unfiltered_count} without filters, deduplicated to {deduplicated_count}")
        
        # Step 3: Rerank the results if reranker is enabled
        if self.use_reranker and deduplicated_count > 0:
            print("Reranking results...")
            final_results = self.reranker.rerank(query, retriever_results, top_k=top_n_from_k)
            print(f"Reranked to {len(final_results)} final results")
        else:
            # If reranker is disabled or no results, use deduplicated results
            print("Skipping reranking, using deduplicated results...")
            final_results = retriever_results.get("deduplicated_results", [])[:top_n_from_k]
        
        # Step 4: Format the results
        print("Formatting results...")
        formatted_results = self.formatter.format_results(final_results, query, metadata)
        print(f"Formatted results: {json.dumps(formatted_results, indent=2)}")
        
        end_time = time.time()
        print(f"Query processing completed in {end_time - start_time:.2f} seconds")
        
        return formatted_results
    
    def _ensure_none_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to convert null/undefined values to None"""
        if not isinstance(data, dict):
            return data
            
        result = {}
        for key, value in data.items():
            if value == "null" or value == "undefined" or value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._ensure_none_values(value)
            elif isinstance(value, list):
                result[key] = [
                    self._ensure_none_values(item) if isinstance(item, dict) else
                    None if item == "null" or item == "undefined" else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
            
    def save_results(self, results: Any, output_path: str):
        """
        Save results to a file.
        
        Args:
            results: The output from process_query
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            if isinstance(results, str):
                f.write(results)
            else:
                json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
# Test the manager if run directly
if __name__ == "__main__":
    # Instantiate the manager
    manager = AgentManager(use_reranker=True, use_llm_explanations=True)
    
    # Define parameters
    query = "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    top_k = 20
    top_n_from_k = 10
    
    # Get results as text
    text_results, json_results = manager.process_query(query, top_k=top_k, top_n_from_k=top_n_from_k, format_as_text=True)
    print("\nText Results:")
    print(text_results)
    
    print("\nJSON Results (sample):")
    if json_results and "recommendations" in json_results and json_results["recommendations"]:
        first_result = json_results["recommendations"][0]
        print(f"Top result: {first_result['assessment_name']}")
        print(f"Reason: {first_result['explanation']}")
        print(f"URL: {first_result.get('assessment_url', 'N/A')}")
    else:
        print("No results found.")