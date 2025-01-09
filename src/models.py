"""
Models module for the sentiment analysis project.
"""

import argparse
from llama_cpp import Llama
from src.config import logger, n_ctx


def load_model(size: str) -> Llama:
    """
    Load the Qwen model based on the size specification with increased context window.

    Args:
        size (str): Model size ('0.5B' or '1.5B')

    Returns:
        Llama: Loaded model instance with increased context window
    """
    model_repo = {
        "0.5B": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "1.5B": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    }.get(size)

    if not model_repo:
        raise ValueError("Invalid model size. Choose either '0.5B' or '1.5B'.")

    # Load model using the recommended quantization file with increased context length
    logger.info(f"Loading model {model_repo} with context window {n_ctx}")
    model = Llama.from_pretrained(
        repo_id=model_repo,
        filename="*Q5_K_M.gguf",
        verbose=False,
        n_ctx=n_ctx,  # Increased context window
    )
    logger.info(f"Model loaded successfully")
    return model


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
        help="Model size to load (0.5B or 1.5B)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.size)
    print(model)
