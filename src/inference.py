"""
Inference module for the sentiment analysis project.
"""

from src.models import load_model
from src.config import CLASSIFIER_PROMPT, TEMPERATURE, USER_PROMPT, logger


def run_inference(model, messages):
    """
    Run inference using the model and predefined test messages.

    Args:
        model: The loaded model instance
        messages (list): List of test messages to process

    Returns:
        list: List of dictionaries containing input messages and model predictions
    """
    results = []
    for message in messages:
        response = model.create_chat_completion(
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(review=message)},
            ],
        )
        prediction = response["choices"][0]["message"]["content"]
        results.append({"input": message, "prediction": prediction})
    return results


if __name__ == "__main__":
    # Predefined test messages
    test_messages = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "What a waste of time. Terrible acting, boring story, I couldn't wait for it to end.",
        "It was okay, not great but not terrible either. Some good moments but overall pretty average.",
    ]

    model = load_model("1.5B")
    results = run_inference(model, test_messages)

    # Print results in a readable format
    logger.info("\nTest Results:")
    for i, result in enumerate(results, 1):
        logger.info(f"\nTest {i}:")
        logger.info(f"Input: {result['input']}")
        logger.info(f"Response: {result['prediction']}\n")
