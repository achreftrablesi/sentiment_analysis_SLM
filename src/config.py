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
CLASSIFIER_PROMPT = """
You are a movie review classifier. 
You will be given a movie review delimited by <review> tags and you will need to classify it as positive or negative.
you only need to return the classification as a single word, either "positive" or "negative".
Output format:
<classification>
</classification>
"""

USER_PROMPT = """
<review>
{review}
</review>
"""

# Inference Parameters
TEMPERATURE = 0.5
TOP_P = 0.9
TOP_K = 50

# Context Window
n_ctx = 16384
