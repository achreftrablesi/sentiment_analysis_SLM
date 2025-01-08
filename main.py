"""
Main script for sentiment analysis using SLMs
"""

import argparse
from src.models import load_model
from src.inference import run_inference
from src.evaluation import evaluate_results
from data.data_loader import load_dataset_subset
from src.config import logger


def main():
    # Argument parser for CLI inputs
    parser = argparse.ArgumentParser(description="Sentiment Analysis using SLMs")
    parser.add_argument(
        "--model_size",
        choices=["0.5B", "1.5B"],
        default="1.5B",
        help="Choose model size",
    )
    parser.add_argument(
        "--sample_size", type=int, default=10, help="Number of reviews to process"
    )
    args = parser.parse_args()

    logger.info(
        f"Starting analysis with model size: {args.model_size}, sample size: {args.sample_size}"
    )

    # Load balanced dataset subset
    logger.info("Loading dataset...")
    dataset = load_dataset_subset(size=args.sample_size)
    logger.info(f"Loaded {len(dataset)} reviews")

    # Convert dataset to test cases
    test_cases = [
        {
            "input": item["review"],
            "label": "positive" if item["label"] == 1 else "negative",
        }
        for item in dataset
    ]

    # Extract just the input messages for inference
    test_messages = [case["input"] for case in test_cases]
    logger.info(
        f"Average review length: {sum(len(msg) for msg in test_messages) / len(test_messages):.0f} characters"
    )

    # Load selected model
    logger.info(f"Loading {args.model_size} model...")
    model = load_model(args.model_size)
    logger.info("Model loaded successfully")

    # Run inference
    logger.info("Starting inference...")
    inference_results = run_inference(model, test_messages)
    logger.info("Inference completed")

    # Combine inference results with true labels
    evaluation_results = []
    for test_case, inference_result in zip(test_cases, inference_results):
        evaluation_results.append(
            {
                "input": test_case["input"],
                "prediction": inference_result["prediction"],
                "label": test_case["label"],
            }
        )

    # Print detailed results
    print("\nTest Results:")
    for i, result in enumerate(evaluation_results, 1):
        logger.info(f"Processing result {i}/{len(evaluation_results)}")
        print(f"\nTest {i}:")
        print(f"Input: {result['input']}")
        print(f"Prediction: {result['prediction']}")
        print(f"True Label: {result['label']}")
        match = result["prediction"] == result["label"]
        print(f"Correct: {'✓' if match else '✗'}")
        logger.info(f"Prediction {'matched' if match else 'did not match'} true label")

    # Evaluate results
    metrics = evaluate_results(evaluation_results)
    logger.info(f"Final accuracy: {metrics['accuracy']:.2%}")
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")



if __name__ == "__main__":
    main()
