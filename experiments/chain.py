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

def confidence_chain(
    model: Any,
    input_text: str,
    student_prompt: str,
    teacher_prompt: str
) -> str:
    """
    Execute a confidence-based chain: student attempts classification first,
    teacher helps if student is uncertain.

    Args:
        model: The language model instance
        input_text: The review text to analyze
        student_prompt: Prompt for the student's initial classification
        teacher_prompt: Prompt for the teacher's expert guidance

    Returns:
        str: Final classification ('positive' or 'negative')
    """
    # Step 1: Student's attempt
    logger.info("Starting first pass: student classification")
    student_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": student_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.3
    )
    student_result = student_response["choices"][0]["message"]["content"]
    logger.info("Finished first pass: student classification complete")

    # Parse student response
    lines = [line.strip() for line in student_result.split('\n') if line.strip()]
    sentiment = ""
    confidence = 0.0
    uncertainty = ""

    for line in lines:
        if line.lower().startswith("sentiment:"):
            sentiment = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("confidence:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.0
        elif line.lower().startswith("uncertainty:"):
            uncertainty = line.split(":", 1)[1].strip()

    # If student is confident (confidence >= 0.5), return their classification
    if confidence >= 0.5 and sentiment in {"positive", "negative"}:
        logger.info("Student was confident in classification")
        return sentiment

    # Step 2: Teacher's guidance
    logger.info("Starting second pass: teacher guidance")
    teacher_context = f"""
    Review: {input_text}
    
    Student's uncertainty: {uncertainty}
    """
    
    teacher_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": teacher_prompt},
            {"role": "user", "content": teacher_context}
        ],
        temperature=0.0
    )
    final_classification = teacher_response["choices"][0]["message"]["content"].strip().lower()
    logger.info("Finished second pass: teacher guidance complete")

    return final_classification

def sarcasm_chain(
    model: Any,
    input_text: str,
    sarcasm_prompt: str,
    classification_prompt: str
) -> str:
    """
    Execute a sarcasm-aware chain: first detect sarcasm, then classify with context
    only if sarcasm is detected.

    Args:
        model: The language model instance
        input_text: The review text to analyze
        sarcasm_prompt: Prompt for sarcasm detection
        classification_prompt: Prompt for sentiment classification

    Returns:
        str: Final classification ('positive' or 'negative')
    """
    # Step 1: Detect sarcasm
    logger.info("Starting first pass: sarcasm detection")
    sarcasm_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": sarcasm_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.2
    )
    sarcasm_result = sarcasm_response["choices"][0]["message"]["content"]
    logger.info("Finished first pass: sarcasm detection complete")

    # Parse sarcasm analysis
    lines = [line.strip() for line in sarcasm_result.split('\n') if line.strip()]
    is_sarcastic = "no"
    examples = "none"

    for line in lines:
        if line.lower().startswith("is_sarcastic:"):
            is_sarcastic = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("examples:"):
            examples = line.split(":", 1)[1].strip().lower()

    # If no sarcasm detected, classify directly
    if is_sarcastic == "no":
        logger.info("No sarcasm detected, classifying directly")
        direct_response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Classify this movie review as positive or negative. Only respond with 'positive' or 'negative'."},
                {"role": "user", "content": input_text}
            ],
            temperature=0.0
        )
        return direct_response["choices"][0]["message"]["content"].strip().lower()

    # Step 2: Only if sarcasm detected, classify with sarcasm context
    logger.info("Starting second pass: informed classification (sarcasm detected)")
    context = f"""
    Review: {input_text}

    Sarcasm Analysis:
    - Contains Sarcasm: {is_sarcastic}
    Examples: {examples}
    """
    
    classification_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": context}
        ],
        temperature=0.0
    )
    final_classification = classification_response["choices"][0]["message"]["content"].strip().lower()
    logger.info("Finished second pass: classification complete")

    return final_classification

def decomposition_chain(
    model: Any,
    input_text: str,
    extract_prompt: str,
    classification_prompt: str
) -> str:
    """
    Execute a decomposition chain: first extract positive/negative aspects,
    then classify based on the decomposition.

    Args:
        model: The language model instance
        input_text: The review text to analyze
        extract_prompt: Prompt for extracting positive/negative aspects
        classification_prompt: Prompt for classifying based on aspects

    Returns:
        str: Final classification ('positive' or 'negative')
    """
    # Step 1: Extract aspects
    logger.info("Starting first pass: aspect extraction")
    extract_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": extract_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.2
    )
    aspects = extract_response["choices"][0]["message"]["content"]
    logger.info("Finished first pass: aspect extraction complete")

    # Step 2: Classify based on extracted aspects
    logger.info("Starting second pass: aspect-based classification")
    classification_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": aspects}
        ],
        temperature=0.0
    )
    final_classification = classification_response["choices"][0]["message"]["content"].strip().lower()
    logger.info("Finished second pass: classification complete")

    return final_classification

def decomposition_deterministic_chain(
    model: Any,
    input_text: str,
    extract_prompt: str
) -> str:
    """
    Execute a deterministic decomposition chain: extract positive/negative aspects,
    then classify based on simple counting of aspects.

    Args:
        model: The language model instance
        input_text: The review text to analyze
        extract_prompt: Prompt for extracting positive/negative aspects

    Returns:
        str: Final classification ('positive' or 'negative')
    """
    # Step 1: Extract aspects
    logger.info("Starting aspect extraction")
    extract_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": extract_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.2
    )
    aspects = extract_response["choices"][0]["message"]["content"]
    logger.info("Finished aspect extraction")

    # Parse aspects and count
    positive_count = 0
    negative_count = 0

    # Split into lines and process each line
    for line in aspects.lower().split('\n'):
        line = line.strip()
        if line.startswith('positive:'):
            # Get everything after "positive:" and split by commas
            aspects_list = line[len('positive:'):].strip().split(',')
            # Count non-empty aspects
            positive_count = sum(1 for aspect in aspects_list if aspect.strip())
        elif line.startswith('negative:'):
            # Get everything after "negative:" and split by commas
            aspects_list = line[len('negative:'):].strip().split(',')
            # Count non-empty aspects
            negative_count = sum(1 for aspect in aspects_list if aspect.strip())

    logger.info(f"Counted aspects - Positive: {positive_count}, Negative: {negative_count}")
    
    # Determine sentiment based on counts
    if positive_count == negative_count:
        # In case of tie, default to negative
        return "negative"
    return "positive" if positive_count > negative_count else "negative"

def star_rating_chain(
    model: Any,
    input_text: str,
    rating_prompt: str,
    resolution_prompt: str
) -> str:
    """
    Execute a two-step chain: first assign a star rating, then resolve mixed ratings.

    Args:
        model: The language model instance
        input_text: The review text to analyze
        rating_prompt: Prompt for initial star rating classification
        resolution_prompt: Prompt for resolving mixed ratings

    Returns:
        str: Final classification ('positive' or 'negative')
    """
    # Step 1: Get star rating
    logger.info("Starting first pass: star rating classification")
    rating_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": rating_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.2
    )
    rating_result = rating_response["choices"][0]["message"]["content"]
    logger.info(f"Star rating result: {rating_result}")
    logger.info("Finished first pass: star rating assigned")

    # Parse rating response
    lines = [line.strip() for line in rating_result.split('\n') if line.strip()]
    rating = None
    sentiment = None

    for line in lines:
        line = line.lower()
        # Handle rating line
        if "rating:" in line:
            try:
                # Remove any commas and get the first number
                rating_part = line.split("rating:")[1].replace(",", "").strip()
                rating = int(rating_part.split()[0])  # Take first word and convert to int
                if not 1 <= rating <= 5:
                    rating = None
            except (ValueError, IndexError):
                logger.warning(f"Could not parse rating from line: {line}")
        
        # Handle sentiment line
        elif "sentiment:" in line:
            # Clean up the sentiment value
            sentiment_part = (
                line.split("sentiment:")[1]
                .replace(",", "")  # Remove commas
                .replace('"', "")  # Remove quotes
                .strip()
            )
            if sentiment_part in {"positive", "negative"}:
                sentiment = sentiment_part

    # If we got a valid rating and it's not 3 stars, use the sentiment
    if rating is not None and rating != 3 and sentiment in {"positive", "negative"}:
        logger.info(f"Clear rating ({rating} stars), returning {sentiment}")
        return sentiment

    # Step 2: Resolve mixed (3-star) or unclear ratings
    logger.info("Starting second pass: resolving mixed rating")
    resolution_context = f"""
    Review: {input_text}

    Initial Analysis:
    This review was rated as {'3 stars' if rating == 3 else 'unclear'}, indicating a mixed sentiment.
    Please analyze whether this mixed review leans more positive or negative.
    """
    
    resolution_response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": resolution_prompt},
            {"role": "user", "content": resolution_context}
        ],
        temperature=0.0
    )
    final_sentiment = resolution_response["choices"][0]["message"]["content"].strip().lower()
    logger.info("Finished second pass: mixed rating resolved")

    # Ensure final sentiment is valid
    if final_sentiment not in {"positive", "negative"}:
        # Default to negative if we can't determine sentiment
        logger.warning(f"Invalid final sentiment: {final_sentiment}, defaulting to negative")
        return "negative"

    return final_sentiment

