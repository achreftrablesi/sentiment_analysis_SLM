"""
Configurations for different experiment types.
"""
from typing import Dict

# Prompt experiment configurations
PROMPT_EXPERIMENTS: Dict[str, Dict] = {
    "zero_shot": {
        "system": """
        You are a movie review classifier. Classify the movie review as positive or negative. 
        Only respond with 'positive' or 'negative'.
        """,
        "description": "Zero-shot classification prompt",
    },
    "one_shot": {
        "system": """
        You are a movie review classifier. Classify the movie review as positive or negative. 
        Only respond with 'positive' or 'negative'.
        Here are some examples:
        - 'I adore this film; it's an absolute classic in my book. 
        Debbie Reynolds as Kathy Selden is charming and full of life. Her witty banter and heartfelt singing bring such joy to the screen.
        The storyline about her budding romance with a silent film star is captivating and perfectly balances humor with sincerity.
        The musical numbers are timeless, especially the iconic "Singin' in the Rain" scene, which is sheer cinematic magic. The supporting cast,
        including Donald O'Connor, adds a layer of fun and camaraderie that’s irresistible. 
        They truly don’t make musicals like this anymore—it's a treasure that never gets old!' -> 'positive'
        - 'I really wanted to enjoy this movie, but it just didn’t work for me. The lead character, played by John Doe, felt one-dimensional and lacked any real depth. 
        The storyline was predictable, with no surprises or emotional weight to keep me engaged. 
        The attempts at humor fell flat, and the dramatic moments felt forced rather than genuine. Even the cinematography, which could have been a saving grace, was uninspired and bland.
        It’s a shame because the premise had so much potential, but the execution left a lot to be desired. Definitely not something I’d watch again.' -> 'negative'
        """,
        "description": "One-shot classification prompt",
    },
    "few_shot": {
        "system": """
        You are a movie review classifier. Classify the movie review as positive or negative. 
        Only respond with 'positive' or 'negative'.
        Here are some examples:
        
        Example 1:
        - 'This movie was fantastic!' -> 'positive'
        - 'Terrible waste of time.' -> 'negative'
        
        Example 2:
        - 'I adore this film; it's an absolute classic in my book. 
        Debbie Reynolds as Kathy Selden is charming and full of life. Her witty banter and heartfelt singing bring such joy to the screen.
        The storyline about her budding romance with a silent film star is captivating and perfectly balances humor with sincerity.
        The musical numbers are timeless, especially the iconic "Singin' in the Rain" scene, which is sheer cinematic magic. The supporting cast,
        including Donald O'Connor, adds a layer of fun and camaraderie that’s irresistible. 
        They truly don’t make musicals like this anymore—it's a treasure that never gets old!' -> 'positive'
        - 'I really wanted to enjoy this movie, but it just didn’t work for me. The lead character, played by John Doe, felt one-dimensional and lacked any real depth. 
        The storyline was predictable, with no surprises or emotional weight to keep me engaged. 
        The attempts at humor fell flat, and the dramatic moments felt forced rather than genuine. Even the cinematography, which could have been a saving grace, was uninspired and bland.
        It’s a shame because the premise had so much potential, but the execution left a lot to be desired. Definitely not something I’d watch again.' -> 'negative'

        Example 3:
        - 'I found this film disappointing on several levels. The plot meandered without much direction, making it hard to care about the outcome.' -> 'negative'
        - 'This movie was an absolute delight from start to finish! The story was engaging, with just the right mix of humor and heartfelt moments to keep me hooked. The lead actor delivered a standout performance, bringing charm and depth to their character..' -> 'positive'

        """,
        "description": "Few-shot classification prompt",
    },
    "CoT": {
        "system": """
        You are a movie review classifier. Analyze movie reviews by following these steps:
        1. Identify key emotional words and phrases
        2. Consider the overall tone and context
        3. Determine if criticism is constructive or harsh
        4. Conclude with ONLY 'positive' or 'negative'

        Remember to ONLY output 'positive' or 'negative' as your final answer.
        """,
        "description": "Chain of Thought classification prompt that guides the model through reasoning steps",
    },
    "CoT_few_shot": {
        "system": """
        You are a movie review classifier. Analyze movie reviews by following these steps:
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
        """,
        "description": "Chain of Thought with Few-Shot examples showing the reasoning process",
    },
}

# Inference parameter configurations :

INFERENCE_EXPERIMENTS: Dict[str, Dict] = {
    "default": {
        "temperature": 0.2,
        "description": "Default inference parameters",
    },
    "strict": {
        "temperature": 0.0,
        "description": "More deterministic parameters",
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
