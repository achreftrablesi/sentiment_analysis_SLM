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
    lines = [line.strip() for line in aspects.split('\n') if line.strip()]
    positive_count = 0
    negative_count = 0

    for line in lines:
        if line.lower().startswith("positive:"):
            # Split the line and count non-empty items after "Positive:"
            items = [item.strip() for item in line.split(":", 1)[1].split(",")]
            positive_count = sum(1 for item in items if item)
        elif line.lower().startswith("negative:"):
            # Split the line and count non-empty items after "Negative:"
            items = [item.strip() for item in line.split(":", 1)[1].split(",")]
            negative_count = sum(1 for item in items if item)

    # Determine sentiment based on counts
    logger.info(f"Counted aspects - Positive: {positive_count}, Negative: {negative_count}")
    return "positive" if positive_count >= negative_count else "negative"

