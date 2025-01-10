"""
Evaluation module for the sentiment analysis project.
"""
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import logger


@dataclass
class PredictionResult:
    """Data class to store prediction results with timing information."""

    input_text: str
    true_label: str
    predicted_label: str
    response_time: float  # in seconds


def evaluate_model_performance(results: List[PredictionResult]) -> Dict:
    """
    Calculate comprehensive evaluation metrics including timing statistics.

    Args:
        results: List of PredictionResult objects containing predictions and timing info

    Returns:
        Dict containing various evaluation metrics
    """
    # Extract predictions and labels
    predictions = [r.predicted_label for r in results]
    true_labels = [r.true_label for r in results]
    response_times = [r.response_time for r in results]

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        true_labels, predictions, labels=["negative", "positive"]
    ).ravel()

    # Timing statistics
    avg_response_time = mean(response_times)
    std_response_time = stdev(response_times) if len(response_times) > 1 else 0
    max_response_time = max(response_times)
    min_response_time = min(response_times)

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "timing": {
            "average_response_time": round(avg_response_time, 3),
            "std_response_time": round(std_response_time, 3),
            "max_response_time": round(max_response_time, 3),
            "min_response_time": round(min_response_time, 3),
            "total_inference_time": round(sum(response_times), 3),
        },
    }

    return metrics


def print_evaluation_report(metrics: Dict) -> None:
    """
    Print a detailed evaluation report.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    logger.info("\n=== Model Evaluation Report ===")

    # Classification metrics
    logger.info("\nClassification Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"True Positives: {metrics['true_positives']}")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    logger.info(f"False Negatives: {metrics['false_negatives']}")

    # Timing metrics
    logger.info("\nTiming Metrics:")
    logger.info(
        f"Average Response Time: {metrics['timing']['average_response_time']:.3f} seconds"
    )
    logger.info(
        f"Response Time Std Dev: {metrics['timing']['std_response_time']:.3f} seconds"
    )
    logger.info(
        f"Fastest Response: {metrics['timing']['min_response_time']:.3f} seconds"
    )
    logger.info(
        f"Slowest Response: {metrics['timing']['max_response_time']:.3f} seconds"
    )
    logger.info(
        f"Total Inference Time: {metrics['timing']['total_inference_time']:.3f} seconds"
    )


if __name__ == "__main__":
    # Example usage with timing information
    test_results = [
        PredictionResult(
            input_text="This movie was fantastic!",
            true_label="positive",
            predicted_label="positive",
            response_time=0.5,
        ),
        PredictionResult(
            input_text="Terrible waste of time.",
            true_label="negative",
            predicted_label="negative",
            response_time=0.4,
        ),
        PredictionResult(
            input_text="I hated this movie.",
            true_label="negative",
            predicted_label="positive",
            response_time=0.6,
        ),
    ]

    # Calculate metrics
    evaluation_metrics = evaluate_model_performance(test_results)

    # Print detailed report
    print_evaluation_report(evaluation_metrics)
