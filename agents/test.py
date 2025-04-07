import argparse
from benchmark import evaluate_system, benchmark_data, evaluate_benchmark, run_benchmark_queries
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation system performance.")
    parser.add_argument("--results", required=False, help="Path to the JSON file with system results.")
    parser.add_argument("--ground_truth", required=False, help="Path to the JSON file with ground truth data.")
    parser.add_argument("--k", type=int, default=3, help="Value of K for Recall@K and MAP@K.")
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration of results")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    ground_truth_path = args.ground_truth or "agents/ground_truth.json"
    results_path = args.results or "system_results.json"
    
    # Create ground truth if needed
    if not os.path.exists(ground_truth_path):
        print(f"Creating benchmark dataset at {ground_truth_path}...")
        with open(ground_truth_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        print("Benchmark dataset created")
    
    # Generate results if needed or requested
    if args.regenerate or not os.path.exists(results_path):
        print(f"Generating system results at {results_path}...")
        run_benchmark_queries(ground_truth_path, results_path, top_k=args.k)
    
    # Evaluate the benchmark
    metrics = evaluate_benchmark(results_path, ground_truth_path, k=args.k)
    
    # Print the results
    print(f"Evaluation Metrics for K={args.k}:")
    print(f"Mean Recall@{args.k}: {metrics[f'recall@{args.k}']:.4f}")
    print(f"Mean Average Precision@{args.k}: {metrics[f'map@{args.k}']:.4f}")
    print(f"Number of Queries Evaluated: {metrics['num_queries']}")

if __name__ == "__main__":
    main()