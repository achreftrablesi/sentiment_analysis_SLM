"""
Models module for the sentiment analysis project.
"""

import argparse

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.config import MODEL_FILENAME, MODEL_REPO, logger, n_ctx


def download_model_file(repo_id: str, filename: str) -> str:
    """
    Download a specific model file from Hugging Face.

    Args:
        repo_id (str): The Hugging Face repository ID.
        filename (str): The name of the file to download.

    Returns:
        str: Local path to the downloaded file.
    """
    try:
        logger.info(f"Downloading {filename} from {repo_id}")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"File downloaded to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise


def load_model(size: str) -> Llama:
    """
    Load the Qwen model based on the size specification with increased context window.

    Args:
        size (str): Model size ('0.5B' or '1.5B')

    Returns:
        Llama: Loaded model instance with increased context window
    """
    model_repo = MODEL_REPO.get(size)
    filename = MODEL_FILENAME.get(size)

    if not model_repo or not filename:
        raise ValueError("Invalid model size. Choose either '0.5B' or '1.5B'.")

    model_file_path = download_model_file(model_repo, filename)

    # Load the model using the downloaded file
    logger.info(f"Loading model from {model_file_path} with context window {n_ctx}")
    try:
        model = Llama(model_path=model_file_path, verbose=False, n_ctx=n_ctx)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Load and initialize Qwen model for sentiment analysis"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["0.5B", "1.5B"],
        default="0.5B",
        help="Model size to load (0.5B or 1.5B)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        model = load_model(args.size)
        logger.info(model)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
