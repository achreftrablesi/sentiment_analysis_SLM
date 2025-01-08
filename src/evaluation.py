"""
Evaluation module for the sentiment analysis project.
"""

from sklearn.metrics import accuracy_score
from src.config import logger


def evaluate_results(results):
    """
    Evaluate model predictions against ground truth labels.

    Args:
        results (list): List of dictionaries containing predictions and labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """

    predictions = [r["prediction"] for r in results]
    labels = [r["label"] for r in results]
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


if __name__ == "__main__":
    # Predefined test cases with known labels
    test_cases = [
        {
            "input": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "prediction": "positive",
            "label": "positive",
        },
        {
            "input": "What a waste of time. Terrible acting, boring story, I couldn't wait for it to end.",
            "prediction": "negative",
            "label": "negative",
        },
        {
            "input": "It was okay, not great but not terrible either. Some good moments but overall pretty average.",
            "prediction": "neutral",
            "label": "neutral",
        },
    ]

    # Test the evaluation function
    metrics = evaluate_results(test_cases)

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")

    # Print detailed comparison
    logger.info("\nDetailed Comparison:")
    for i, case in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}:")
        logger.info(f"Input: {case['input']}")
        logger.info(f"Prediction: {case['prediction']}")
        logger.info(f"True Label: {case['label']}")
        logger.info(f"Correct: {'✓' if case['prediction'] == case['label'] else '✗'}")
