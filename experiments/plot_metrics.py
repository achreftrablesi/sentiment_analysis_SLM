"""
Visualization utilities for experiment results.
"""
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict


def plot_accuracy_comparison(results: Dict, experiment_name: str, output_dir: Path):
    """Plot accuracy comparison between models."""
    plt.figure(figsize=(8, 6))
    accuracies = [results[size]["accuracy"] for size in ["0.5B", "1.5B"]]
    
    plt.bar(["0.5B Model", "1.5B Model"], accuracies)
    plt.title(f"Accuracy Comparison\n{experiment_name}")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, v in enumerate(accuracies):
        plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png")
    plt.close()


def plot_timing_metrics(results: Dict, experiment_name: str, output_dir: Path):
    """Plot timing metrics comparison."""
    plt.figure(figsize=(10, 6))
    timing_metrics = ["average_response_time", "max_response_time", "min_response_time"]
    model_sizes = ["0.5B", "1.5B"]
    
    x = range(len(timing_metrics))
    width = 0.35
    
    for i, size in enumerate(model_sizes):
        times = [results[size]["timing"][metric] for metric in timing_metrics]
        plt.bar([xi + i*width for xi in x], times, width, label=f"{size} Model")

    plt.xlabel("Metric")
    plt.ylabel("Time (seconds)")
    plt.title(f"Response Time Comparison\n{experiment_name}")
    plt.xticks([xi + width/2 for xi in x], [m.replace("_", " ").title() for m in timing_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "timing_comparison.png")
    plt.close()


def create_experiment_visualizations(results_path: Path):
    """Create all visualizations for an experiment."""
    with open(results_path / "results.json") as f:
        data = json.load(f)
    
    experiment_name = data["experiment_name"]
    results = data["results"]
    
    plot_accuracy_comparison(results, experiment_name, results_path)
    plot_timing_metrics(results, experiment_name, results_path)