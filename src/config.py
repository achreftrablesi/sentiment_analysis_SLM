"""
Configuration file for the sentiment analysis project.
"""

import logging

# Dataset Path
DATASET_PATH = "ajaykarthick/imdb-movie-reviews"

# Model Repos
MODEL_REPO = {
    "0.5B": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    "1.5B": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sentiment_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Classifier Prompt
CLASSIFIER_PROMPT_1_5B = """
You are a movie review classifier. 
You will be given a movie review delimited by <review> tags. Classify it as positive or negative.
Only respond with 'positive' or 'negative'.
"""

CLASSIFIER_PROMPT_0_5B = """
You are a movie review classifier. Analyze movie reviews delimited by <review> tags by following these steps:
1. Identify key emotional words and phrases
2. Consider the overall tone and context
3. Determine if criticism is constructive or harsh
4. Conclude with ONLY 'positive' or 'negative'

Here are some examples:

Review: "While the special effects were good, the plot was confusing and the acting was terrible. I wouldn't recommend it."
Steps:
1. Key phrases: "special effects were good" (+), "plot was confusing" (-), "acting was terrible" (-), "wouldn't recommend" (-)
2. Tone: Mostly critical
3. Criticism: Harsh and definitive
Answer: negative

Review: "Despite some pacing issues in the middle, the strong performances and beautiful cinematography make this film a must-watch!"
Steps:
1. Key phrases: "pacing issues" (-), "strong performances" (+), "beautiful cinematography" (+), "must-watch" (+)
2. Tone: Generally enthusiastic
3. Criticism: Constructive, outweighed by positives
Answer: positive

Now analyze the given review following the same steps and respond ONLY with 'positive' or 'negative'.
"""

USER_PROMPT = """
<review>
{review}
</review>
"""

# Inference Parameters
TEMPERATURE = 0.2
MAX_TOKENS = 128


# Context Window
n_ctx = 16384