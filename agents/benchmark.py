import json
from typing import Dict, List
import numpy as np
import os
from agent_manager import AgentManager 

def calculate_recall_at_k(retrieved_items: List[str], 
                          relevant_items: List[str], 
                          k: int = 3) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        retrieved_items: List of IDs/names of retrieved items
        relevant_items: List of IDs/names of relevant items (ground truth)
        k: Cut-off for retrieval
        
    Returns:
        Recall@K score (0.0-1.0)
    """
    if not relevant_items:
        return 1.0  # If no relevant items exist, return perfect score
    
    # Limit to top-k items
    retrieved_items_at_k = retrieved_items[:k]
    
    # Count relevant items in the top-k
    relevant_retrieved = [item for item in retrieved_items_at_k if item in relevant_items]
    
    # Calculate Recall@K
    recall = len(relevant_retrieved) / len(relevant_items)
    return recall

def calculate_map_at_k(retrieved_items: List[str], 
                       relevant_items: List[str], 
                       k: int = 3) -> float:
    """
    Calculate Mean Average Precision at K (MAP@K).
    
    Args:
        retrieved_items: List of IDs/names of retrieved items
        relevant_items: List of IDs/names of relevant items (ground truth)
        k: Cut-off for retrieval
        
    Returns:
        MAP@K score (0.0-1.0)
    """
    if not relevant_items:
        return 1.0  # If no relevant items exist, return perfect score
    
    # Limit to top-k
    retrieved_at_k = retrieved_items[:k]
    
    # Calculate precision at each position where a relevant item is retrieved
    precisions = []
    num_relevant_seen = 0
    
    for i, item in enumerate(retrieved_at_k):
        position = i + 1  # 1-based position
        if item in relevant_items:
            num_relevant_seen += 1
            # Precision at this position = relevant seen / position
            precisions.append(num_relevant_seen / position)
    
    # Average precision is the average of precision values at relevant items
    if precisions:
        average_precision = sum(precisions) / len(relevant_items)
    else:
        average_precision = 0.0
        
    return average_precision

def evaluate_benchmark(results_file: str, ground_truth_file: str, k: int = 3) -> Dict[str, float]:
    """
    Evaluate system performance using a benchmark dataset.
    
    Args:
        results_file: Path to JSON file with system results
        ground_truth_file: Path to JSON file with ground truth data
        k: Cut-off for retrieval
        
    Returns:
        Dictionary with metrics (Recall@K, MAP@K)
    """
    # Load results and ground truth
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Calculate metrics for each query
    recall_scores = []
    map_scores = []
    
    for query_id in ground_truth:
        if query_id in results:
            # Extract assessment names/IDs
            retrieved = [item["name"] for item in results[query_id]["assessments"]]
            relevant = ground_truth[query_id]["relevant_assessments"]
            
            # Calculate metrics
            recall = calculate_recall_at_k(retrieved, relevant, k=k)
            map_score = calculate_map_at_k(retrieved, relevant, k=k)
            
            recall_scores.append(recall)
            map_scores.append(map_score)
    
    # Calculate average metrics across all queries
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_map = np.mean(map_scores) if map_scores else 0.0
    
    return {
        f"recall@{k}": avg_recall,
        f"map@{k}": avg_map,
        "num_queries": len(recall_scores)
    }

# Create benchmark_dataset.json
benchmark_data = {
    "cashier_query": {
        "query_text": "Need entry-level cashier assessments evaluating customer service and accuracy",
        "relevant_assessments": [
            "Cashier Solution",  # From pre-packaged solutions
            "Customer Service Assessment",  # Individual test
            "Accuracy and Speed Test"  # Individual test
        ]
    },
    "manager_query": {
        "query_text": "Looking for retail store manager assessments with inventory management skills",
        "relevant_assessments": [
            "Store Manager Solution",  # From pre-packaged solutions
            "Inventory Management Simulation",  # Individual test
            "Retail Operations Knowledge Test"  # Individual test
        ]
    },
    "tech_query": {
        "query_text": "Assessments for Python developers with SQL skills",
        "relevant_assessments": [
            "Python (New)",  # Individual test
            "SQL (New)",  # Individual test
            "Developer Aptitude Test"  # From pre-packaged solutions
        ]
    },
    "bilingual_query": {
        "query_text": "Bilingual Spanish customer service assessments",
        "relevant_assessments": [
            "Bilingual Spanish Reservation Agent Solution",  # From pre-packaged solutions
            "Spanish Language Proficiency Test",  # Individual test
            "Customer Interaction Assessment"  # Individual test
        ]
    }
}

with open("benchmark_dataset.json", "w") as f:
    json.dump(benchmark_data, f, indent=2)

# Expected results structure
expected_results = {
    "cashier_query": {
        "assessments": [
            {"name": "Cashier Solution"},
            {"name": "Customer Service Assessment"},
            {"name": "Accuracy and Speed Test"}
        ]
    },
    "manager_query": {
        "assessments": [
            {"name": "Store Manager Solution"},
            {"name": "Inventory Management Simulation"},
            {"name": "Retail Operations Knowledge Test"}
        ]
    },
    "tech_query": {
        "assessments": [
            {"name": "Python (New)"},
            {"name": "SQL (New)"},
            {"name": "Developer Aptitude Test"}
        ]
    },
    "bilingual_query": {
        "assessments": [
            {"name": "Bilingual Spanish Reservation Agent Solution"},
            {"name": "Spanish Language Proficiency Test"},
            {"name": "Customer Interaction Assessment"}
        ]
    }
}

with open("expected_results.json", "w") as f:
    json.dump(expected_results, f, indent=2)

def evaluate_system():
    """
    Generate system results and evaluate performance
    """
    # Define file paths
    benchmark_file = "agents/ground_truth.json"  # Use your actual ground truth path
    results_file = "system_results.json"
    
    # Check if files exist
    if not os.path.exists(benchmark_file):
        print(f"Error: Ground truth file not found at {benchmark_file}")
        return
        
    # Generate results if needed
    if not os.path.exists(results_file):
        print("System results file not found, generating now...")
        run_benchmark_queries(benchmark_file, results_file)
    
    # Calculate metrics
    metrics = evaluate_benchmark(
        results_file=results_file,
        ground_truth_file=benchmark_file,
        k=3
    )
    
    print(f"\nSystem Performance:")
    print(f"Average Recall@3: {metrics['recall@3']:.2f}")
    print(f"Average MAP@3: {metrics['map@3']:.2f}")
    print(f"Queries evaluated: {metrics['num_queries']}")
    
def run_benchmark_queries(benchmark_file: str, output_file: str, top_k: int = 3):
    """
    Run all benchmark queries through the recommendation system and save results
    """
    # Load benchmark queries
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)
    
    # Initialize the agent manager with optional reranker
    # Setting use_reranker=True but with the timeout mechanism we'll add to agent_manager.py
    agent_manager = AgentManager(use_reranker=True, use_llm_explanations=False)
    
    # Process each query and collect results
    results = {}
    for query_id, query_info in benchmark_data.items():
        print(f"Processing benchmark query: {query_id}")
        
        # Get recommendations for this query
        response = agent_manager.process_query(
            query=query_info["query_text"],
            top_k=20,  # Increase to get more candidates
            top_n_from_k=3  # But only return top 3
        )
        
        # Extract assessment names and convert to expected format
        assessments = []
        if response and "recommendations" in response:
            for rec in response["recommendations"]:
                assessments.append({
                    "name": rec.get("assessment_name", "Unknown"),
                    "score": rec.get("score", None)
                })
        
        # Store results for this query
        results[query_id] = {
            "assessments": assessments
        }
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results saved to {output_file}")
    return results