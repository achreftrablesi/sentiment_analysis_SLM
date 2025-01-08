"""
Configurations for different experiment types.
"""
from typing import Dict

# Prompt experiment configurations
PROMPT_EXPERIMENTS: Dict[str, Dict] = {
    "zero_shot": {
        "system": "Classify the movie review as positive or negative. Only respond with 'positive' or 'negative'.",
        "description": "Zero-shot classification prompt",
    },
    "one_shot": {
        "system": """
        Classify the movie review as positive or negative. Only respond with 'positive' or 'negative'.
        Here are some examples:
        - 'This movie was fantastic!' -> 'positive'
        - 'Terrible waste of time.' -> 'negative'
        """,
        "description": "One-shot classification prompt",
    },
    "few_shot": {
        "system": """
        Classify the movie review as positive or negative. Only respond with 'positive' or 'negative'.
        Here are some examples:
        
        Example 1:
        - 'This movie was fantastic!' -> 'positive'
        - 'Terrible waste of time.' -> 'negative'
        
        Example 2:
        - 'the story was good but the acting was bad.' -> 'negative'
        - 'The movie was great, I loved the plot and the acting was amazing.' -> 'positive'
        
        Example 3:
        - 'The movie was good, but the ending was disappointing.' -> 'negative'
        - 'I loved the movie, the acting was amazing and the plot was thrilling.' -> 'positive'

        """,
        "description": "Few-shot classification prompt",
    },
}

# Inference parameter configurations : 

INFERENCE_EXPERIMENTS: Dict[str, Dict] = {
    "default": {
        "temperature": 0.5,
        "description": "Default inference parameters",
    },
    "strict": {
        "temperature": 0.0,
        "description": "More deterministic parameters",
    },
    "creative": {
        "temperature": 1.0,
        "description": "More variable parameters",
    },
}


def get_experiment_config(experiment_type: str, experiment_name: str) -> Dict:
    """
    Get configuration for a specific experiment.

    Args:
        experiment_type: Type of experiment ('prompt' or 'params')
        experiment_name: Name of the specific experiment configuration

    Returns:
        Dict containing the experiment configuration

    Raises:
        ValueError: If invalid experiment type or name is provided
    """
    if experiment_type == "prompt":
        if experiment_name not in PROMPT_EXPERIMENTS:
            raise ValueError(
                f"Unknown prompt experiment: {experiment_name}. "
                f"Available options: {list(PROMPT_EXPERIMENTS.keys())}"
            )
        return PROMPT_EXPERIMENTS[experiment_name]

    elif experiment_type == "params":
        if experiment_name not in INFERENCE_EXPERIMENTS:
            raise ValueError(
                f"Unknown parameter experiment: {experiment_name}. "
                f"Available options: {list(INFERENCE_EXPERIMENTS.keys())}"
            )
        return INFERENCE_EXPERIMENTS[experiment_name]

    else:
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. "
            "Available options: 'prompt' or 'params'"
        )
