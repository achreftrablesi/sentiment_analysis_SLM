"""
Main module for sentiment analysis inference.
"""
import argparse
from time import time
from src.models import load_model
from src.config import CLASSIFIER_PROMPT, USER_PROMPT, logger


def analyze_sentiment(review_text: str, model_size: str = "1.5B") -> dict:
    """
    Analyze the sentiment of a single movie review.

    Args:
        review_text: The movie review text to analyze
        model_size: Size of the model to use ('0.5B' or '1.5B')

    Returns:
        Dict containing the prediction and response time
    """
    # Load model
    logger.info(f"Loading {model_size} model...")
    model = load_model(model_size)

    # Run inference with timing
    start_time = time()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(review=review_text)},
        ],
    )
    inference_time = time() - start_time

    prediction = response["choices"][0]["message"]["content"].strip().lower()

    return {
        "review": review_text,
        "prediction": prediction,
        "inference_time": inference_time
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment of a movie review")
    parser.add_argument("review", type=str, help="Movie review text to analyze")
    parser.add_argument(
        "--model_size",
        choices=["0.5B", "1.5B"],
        default="1.5B",
        help="Choose model size",
    )
    args = parser.parse_args()

    # Run analysis
    result = analyze_sentiment(args.review, args.model_size)

    # Print results
    logger.info("\nSentiment Analysis Results:")
    logger.info(f"Review: {result['review']}")
    logger.info(f"Sentiment: {result['prediction']}")
    logger.info(f"Processing Time: {result['inference_time']:.3f} seconds")


if __name__ == "__main__":
    main()
