"""
Chain prompting strategies for sentiment analysis.
"""
from typing import Any
from src.config import logger

def summary_chain(
    model: Any,
    input_text: str,
    summary_prompt: str,
    classification_prompt: str
) -> str:
    """
    Execute a two-step chain: first summarize the review, then classify the sentiment.

    Args:
        model: The language model instance
        input_text: The review text to analyze
        summary_prompt: Prompt for generating the summary
        classification_prompt: Prompt for classifying the sentiment

    Returns:
        str: Final classification result in the format:
            Sentiment: [positive/negative]
            Confidence: [0.0-1.0]
            Explanation: [Brief explanation]
    """
    # Step 1: Generate summary
    logger.info("Starting first pass: creating summary")
    first_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.2
    )
    summary = first_response["choices"][0]["message"]["content"]
    logger.info("Finished first pass: summary created")

    # Step 2: Classify sentiment based on summary
    logger.info("Starting second pass: classifying")
    second_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": summary}
        ],
        temperature=0.0
    )
    logger.info("Finished second pass: classification complete")
    
    return second_response["choices"][0]["message"]["content"]

