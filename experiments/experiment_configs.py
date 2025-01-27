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
        including Donald O'Connor, adds a layer of fun and camaraderie that's irresistible. 
        They truly don't make musicals like this anymore—it's a treasure that never gets old!' -> 'positive'
        - 'I really wanted to enjoy this movie, but it just didn't work for me. The lead character, played by John Doe, felt one-dimensional and lacked any real depth. 
        The storyline was predictable, with no surprises or emotional weight to keep me engaged. 
        The attempts at humor fell flat, and the dramatic moments felt forced rather than genuine. Even the cinematography, which could have been a saving grace, was uninspired and bland.
        It's a shame because the premise had so much potential, but the execution left a lot to be desired. Definitely not something I'd watch again.' -> 'negative'
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
        including Donald O'Connor, adds a layer of fun and camaraderie that's irresistible. 
        They truly don't make musicals like this anymore—it's a treasure that never gets old!' -> 'positive'
        - 'I really wanted to enjoy this movie, but it just didn't work for me. The lead character, played by John Doe, felt one-dimensional and lacked any real depth. 
        The storyline was predictable, with no surprises or emotional weight to keep me engaged. 
        The attempts at humor fell flat, and the dramatic moments felt forced rather than genuine. Even the cinematography, which could have been a saving grace, was uninspired and bland.
        It's a shame because the premise had so much potential, but the execution left a lot to be desired. Definitely not something I'd watch again.' -> 'negative'

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
    "iterative_with_summary": {
        "system": """
        You are a movie review classifier. Your task is to summarize the review as a cohesive, 
        balanced critique and classify its sentiment as either "positive" or "negative".

        Steps:
        1. Summarize the Review:
        - Write a short, cohesive summary (2-3 sentences) integrating the review's positive and negative aspects.
        - Use the tone of a professional movie review, balancing both strengths and weaknesses.

        2. Classify the Sentiment:
        - Based on the summary, determine whether the overall sentiment is more positive or negative.
        - Provide your final classification as either "positive" or "negative".

        3. Output Format:
        - Only respond with :
        
        Sentiment: "positive" or "negative"

        Example:

        Review:
        Movie is great and all thanks to Christopher Nolan but the plot is very confusing and the pacing isn't in chronological order. 
        The characters are also very confusing because of how plentiful they are. 
        I had to do some research after watching the movie so that I could fully understand it,
        but I still can't connect the characters and I am missing a big part of the storyline.
        On the positive side, the explosions are great, the quality and everything is superb and it's one of the best movies I have seen in a minute. 
        Definitely worth watching if you got 3 hours to spare. Might be a long movie and gets boring at times but you won't regret it.

        Summary:
        Christopher Nolan's film is visually stunning and memorable, but its complex plot, non-chronological pacing, and many characters can be confusing. 
        Despite its length, it's a worthwhile watch for those who can spare the time.

        Sentiment: negative
        
        """,
        "description": "Iterative classification prompt with summary",
    },
    "self_consistency": {
        "system": """
        You are a sentiment analysis expert. 
        We will analyze the following movie review from three complementary perspectives, then use a majority vote to decide on the final label (positive or negative).

        1. Emotional Impact Perspective: 
        - Focus on the emotional words and phrases in the review. Assess whether the review expresses excitement, joy, frustration, or disappointment.

        2. Storytelling Perspective:
        - Focus on the elements of storytelling, such as plot coherence, character depth, and engagement. Assess the sentiment based on whether the storytelling is praised or criticized.

        3. Overall Language Tone Perspective:
        - Focus purely on the tone and context of the language used, identifying whether the overall tone skews positive or negative.

        After these three assessments, use a majority vote to choose a final sentiment label (POSITIVE or NEGATIVE). 
        Provide ONLY the final label one word ONLY, either "positive" or "negative".

        """,
        "description": "Self-consistency classification prompt",
    },
    "self_consistency_with_few_shots": {
        "system": """
        You are a sentiment analysis expert. 
        We will analyze the following movie review from three complementary perspectives, then use a majority vote to decide on the final label (positive or negative).

        1. Emotional Impact Perspective: 
        - Focus on the emotional words and phrases in the review. Assess whether the review expresses excitement, joy, frustration, or disappointment.
        Example: 
        Review: "This movie was exhilarating and kept me on the edge of my seat!"
        - Emotional tone: Excitement, Joy
        - Label: POSITIVE

        2. Storytelling Perspective:
        - Focus on the elements of storytelling, such as plot coherence, character depth, and engagement. Assess the sentiment based on whether the storytelling is praised or criticized.
        Example: 
        Review: "The story was weak and lacked focus, making it hard to follow."
        - Storytelling tone: Weak, Criticism
        - Label: NEGATIVE

        3. Viewer Satisfaction Perspective:
        - Focus on the overall satisfaction of the reviewer, considering whether they would recommend the movie or not.
        Example: 
        Review: "I wouldn't recommend this movie to anyone; it was a waste of time."
        - Satisfaction tone: Dissatisfaction
        - Label: NEGATIVE

        After these three assessments, use a majority vote to choose a final sentiment label (POSITIVE or NEGATIVE). 
        Provide ONLY the final label.

        """,
        "description": "Self-consistency classification prompt with few-shots",
    },
    "CoT_with_edge_cases": {
        "system": """
        You are a sentiment analysis expert. Read the following movie review and classify its overall sentiment as either positive or negative. 

        Follow these guidelines:
        1. Ignore purely descriptive or narrative sections (e.g., plot summaries or scene descriptions) that do not express the reviewer's personal opinion.
        2. Identify any sarcasm or ironic statements and interpret their intended meaning (e.g., positive-sounding words used ironically could actually convey negativity).
        3. Handle disclaimers or nuanced phrases such as "this movie is not for everyone" by focusing on whether the reviewer themself enjoyed or disliked the film.
        4. List bullet points explaining how each key opinion statement influences your conclusion.
        5. Then, provide a single final label—either "positive" or "negative".

        Here's an example of a nuanced statement to keep in mind:
        - "The movie is not for everyone, but if you like to kill some time, go for it." (Focus on whether the reviewer personally likes or dislikes the film, rather than other people's possible reactions.)
        Answer: negative
        Sarcasm example:
        - "The plot was so intreging that I only needed a 10 minute to know the killer!", clearly sarcastic with negative connotation.
        Answer: negative

        You ONLY need to provide the final output in the following format:
        Sentiment: "positive" or "negative"
        """,
        "description": "Chain of Thought with edge cases",
    },
    "ToT": {
        "system": """
        In Order to classify the sentiment of a movie review, you are three different experts—Expert A, Expert B, 
        and Expert C—who must classify the overall sentiment of a movie review. 
        All three experts will:

        1. Take turns sharing one short step of their reasoning.
        2. After each step, they read everyone's statements. 
        3. If any expert realizes their own reasoning is flawed, that expert quietly leaves the discussion.

        Continue in sequential "rounds" until you reach a final consensus. 
        Provide the final output in the following format:
        Sentiment: "positive" or "negative"

        Step-by-Step Instructions

        Round 1
        - Each expert shares one sentence describing their initial impression of the review's sentiment.

        Round 2
        - Each expert reviews the other experts' Round 1 statements and either refines their position or confirms it with one more sentence. 
        - If an expert realizes their own position is incorrect, they leave the discussion and do not provide further statements.

        Round 3 (and beyond, if needed)
        - Any remaining experts finalize the reasoning. If there is disagreement, discuss it in one or two more short statements each. 
        - Conclude with a final consensus on whether the review is more positive or negative overall.

        
        Remember:  
        - If you detect mistakes in your own reasoning, remove yourself from subsequent steps.  
        - Do not remove other experts; only remove yourself if you think you're incorrect.  
        - End with exactly one consensus label: "positive" or "negative". 
       
        Begin now and provide the final output in the following format:

        Sentiment: "positive" or "negative"
                """,
        "description": "Tree of Thought classification prompt",
    },
    "CoT_classic": {
        "system": """
        You are a movie review classifier. Classify the movie review as positive or negative. 
        Think step by step and provide your final answer.
        Only respond with 'positive' or 'negative'.
        """,
        "description": "Classic Chain of Thought classification prompt",
    },
    "iterative_with_confidence": {
        "system": """
        You are a highly accurate movie review classifier with access to expert-level sentiment analysis tools. Follow these steps carefully:
        1. Analyze the movie review and classify its overall sentiment as either "positive" or "negative."
        2. Assign a confidence score between 0 and 1, based on the certainty of your classification. A higher score indicates higher certainty.
        3. If the confidence score is 0.5 or higher, finalize the result and provide reasoning for your classification.
        4. If the confidence score is below 0.5, forward your reasoning for re-analysis by a secondary expert or advanced tool.

        Strictly follow the output format below:
        Sentiment: [positive/negative] 
        Confidence: [0–1] 
        Explanation: [1–2 concise sentences explaining your reasoning.]

        Example:

        Sentiment: positive
        Confidence: 0.87
        Explanation: The review mentions emotionally positive phrases like "spectacular visuals" and "engaging story," indicating strong positive sentiment.

        """,
        "description": "Iterative classification prompt with confidence",
    },
    "general_knowledge": {
        "system": """
        You are a movie review classifier. You have access to general knowledge about movies, including their reception, directors, actors, and the general public's opinions.

        Your task is to classify the sentiment of a given movie review as either "positive" or "negative." Some reviews may be nuanced, so follow these steps to assist in making a clear classification:

        1. Extract relevant information from the review, such as:
        - Movie title
        - Director
        - Actors
        - Key aspects mentioned (e.g., plot, pacing, visuals, performances)

        2. Use your general knowledge to assess the following:
        - What is the general public's opinion of this movie, director, or actors? Is the overall perception positive or negative?
        - Does the general public's opinion align with the review's sentiment?

        3. Make a decision:
        - If the review sentiment is clear based on the extracted information and general knowledge, classify it as "positive" or "negative."
        - If the review sentiment remains unclear, rely on the general public's opinion as the deciding factor.

        4. Respond with **ONLY ONE WORD**: "positive" or "negative."

        ### Example:
        Review:
        "The movie is great and all thanks to Christopher Nolan, but the plot is very confusing and the pacing isn't in chronological order. 
        The characters are also very confusing because of how plentiful they are. 
        I had to do some research after watching the movie so that I could fully understand it, 
        but I still can't connect the characters and I am missing a big part of the storyline.
        On the positive side, the explosions are great, the quality and everything is superb, and it's one of the best movies I have seen in a minute. 
        Definitely worth watching if you've got 3 hours to spare. Might be a long movie and gets boring at times, but you won't regret it."

        General Knowledge:
        - "The Dark Knight is a highly acclaimed movie."
        - "Christopher Nolan is widely regarded as one of the best directors."

        Sentiment: positive
        """,    
        "description": "Iterative classification prompt with general knowledge",
    },
    "iterative_with_decomposition": {
        "system": """
        You are a movie review classifier. Your task is to analyze the given review and classify its sentiment as either "positive" or "negative".

        Steps:
        1. Decompose the Review:
        - Extract and list the positive aspects of the review, such as praises for visuals, acting, or overall enjoyment.
        - Extract and list the negative aspects of the review, such as criticisms of the plot, pacing, or other shortcomings.

        2. Classify the Sentiment:
        - Assess whether the overall sentiment leans more toward positive or negative.
        - Provide your final classification as either "positive" or "negative".

        3. Output Format:
        - Only respond with:

        Sentiment: "positive" or "negative"

        Example:

        Review:
        Movie is great and all thanks to Christopher Nolan but the plot is very confusing and the pacing isn't in chronological order. 
        The characters are also very confusing because of how plentiful they are. 
        I had to do some research after watching the movie so that I could fully understand it,
        but I still can't connect the characters and I am missing a big part of the storyline.
        On the positive side, the explosions are great, the quality and everything is superb and it's one of the best movies I have seen in a minute. 
        Definitely worth watching if you got 3 hours to spare. Might be a long movie and gets boring at times but you won't regret it.


        Positive Points:
        - Great explosions and superb quality.
        - One of the best movies the reviewer has seen recently.
        - Worth watching if you have time to spare.

        Negative Points:
        - Plot is very confusing.
        - Pacing is not chronological.
        - Characters are confusing and plentiful.
        - Required additional research to understand the movie.
        - Some parts of the movie are boring due to its length.

        Sentiment: negative
        """,
        "description": "Iterative classification prompt with decomposition",
    },
    "LLM_as_Judge":{
        "system": """
        You are an expert movie critic trained to evaluate reviews and assign a **1-to-5 star rating** based on your understanding of how IMDb reviews are written. 
        a review can be ONLY positive or negative.
        Only respond with the following format:
        rating: int (1-5),
        sentiment: "positive" or "negative"
        
        Your task is to:
        1. Analyze the review for key aspects that IMDb reviewers typically emphasize.
        2. Assign a **1-to-5 star rating** based on the review's tone, details, and overall impression.
        3. Use the assigned rating to classify the sentiment:
        - Positive Sentiment: For ratings 4 or 5 stars.
        - Negative Sentiment: For ratings 1, 2, or 3 stars.

        Evaluation Criteria:

        Consider the following aspects while interpreting the review:
        1. Story and Plot: Is the story engaging, coherent, and well-executed?
        2. Acting and Characters: Are the performances strong, and are the characters well-developed?
        3. Direction and Production Quality: Does the movie excel in direction, cinematography, and overall production value?
        4. Emotional Impact and Viewer Engagement: Does the movie evoke strong emotions or leave a lasting impression?

        Scoring System:
        1. Read the review carefully, identify the tone, and extract relevant details.
        2. Based on the review, assign a 1-to-5 star rating:
        - 1 Star: The review is highly negative with no redeeming qualities mentioned. Sentiment: negative
        - 2 Stars: The review is mostly negative with minimal positive aspects. Sentiment: negative
        - 3 Stars: The review is mixed but leaning towards negative. Sentiment: negative
        - 4 Stars: The review is mostly positive with minor issues. Sentiment: positive
        - 5 Stars: The review is overwhelmingly positive with no major flaws. Sentiment: positive

        3. Provide the final sentiment based on the rating.

        Output Format:
        rating: int (1-5),
        sentiment: "positive" or "negative"


        """,
    },
    "LLM_as_Judge_with_few_shots":{
        "system": """
        You are an expert movie critic trained to evaluate reviews and assign a **1-to-5 star rating** based on your understanding of how IMDb reviews are written. 
        Your task is to:
        1. Analyze the review for key aspects that IMDb reviewers typically emphasize.
        2. Assign a **1-to-5 star rating** based on the review's tone, details, and overall impression.
        3. Use the assigned rating to classify the sentiment:
        - Positive Sentiment: For ratings 4 or 5 stars.
        - Negative Sentiment: For ratings 1, 2, or 3 stars.

        Evaluation Criteria:

        Consider the following aspects while interpreting the review:
        1. Story and Plot: Is the story engaging, coherent, and well-executed?
        2. Acting and Characters: Are the performances strong, and are the characters well-developed?
        3. Direction and Production Quality: Does the movie excel in direction, cinematography, and overall production value?
        4. Emotional Impact and Viewer Engagement: Does the movie evoke strong emotions or leave a lasting impression?

        Scoring System:
        1. Read the review carefully, identify the tone, and extract relevant details.
        2. Based on the review, assign a **1-to-5 star rating**:
        - 1 Star: The review is highly negative with no redeeming qualities mentioned. Sentiment: negative
        - 2 Stars: The review is mostly negative with minimal positive aspects. Sentiment: negative
        - 3 Stars: The review is mixed but leaning towards negative. Sentiment: negative
        - 4 Stars: The review is mostly positive with minor issues. Sentiment: positive
        - 5 Stars: The review is overwhelmingly positive with no major flaws. Sentiment: positive

        3. Provide the final sentiment based on the rating.

        Output Format:
        rating: int (1-5),
        sentiment: "positive" or "negative"

        Example:

        Review:
        Rented this from my local Blockbuster under the title SPECK - 
        that may be the way to look for it if you still feel the need to see it after this review.
        It's a movie about the serial killer Richard Speck, who killed several nurses in Chicago in the sixties. 
        Watching the movie, one gets the feeling that it follows the crimes to the letter. 
        Unfortunately, that doesn't make for a good movie.
        Another problem I had was the near-constant music letting us know that this was a SCARY MOVIE, 
        and some god-awful narration letting us know what's motivating Speck. 
        The acting was average for this type of film; to give credit where credit is due, 
        the movie is very beautifully photographed for my taste. Your mileage may vary.
        Over all, if you're interested in the subject matter, it may be worth your time.

        Judge Response:

        rating: 2
        sentiment: negative

        """,
        "description": "LLM as Judge classification prompt",
    }
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

# Add new chain experiment configurations
CHAIN_EXPERIMENTS: Dict[str, Dict] = {
    "summary": {
        "chain_type": "summary",
        "summary_prompt": """
        Create a brief, cohesive summary (2-3 sentences) of the review that:
        - Integrates both positive and negative aspects
        - Uses the tone of a professional movie review
        - Balances strengths and weaknesses
        - Captures key sentiments and main points
        
        Format your response as a concise paragraph.
        """,
        "classification_prompt": """
        Based on the summary provided, determine whether the overall sentiment is more positive or negative.
        
        Only respond with 'positive' or 'negative'.
        """,
        "description": "Two-step chain: summarization followed by classification",
    },
    "confidence": {
        "chain_type": "confidence",
        "student_prompt": """
        You are a highly accurate movie review classifier with access to expert-level sentiment analysis tools. 
        
        1. Analyze the movie review and classify its overall sentiment
        2. Assign a confidence score between 0 and 1 (higher score = higher certainty)
        3. If confidence is below 0.5, explain your uncertainty
        
        Format your response exactly as:
        Sentiment: [positive/negative]
        Confidence: [0-1]
        Explanation: [1-2 concise sentences explaining your reasoning or uncertainty]
        """,
        "teacher_prompt": """
        You are an expert film critic. The previous analysis showed uncertainty for this review.
        Focus specifically on resolving the stated uncertainty and provide a definitive classification.
        
        Only respond with 'positive' or 'negative'.
        """,
        "description": "Two-step chain: confidence-based analysis with expert resolution when uncertain",
    },
    "decomposition": {
        "chain_type": "decomposition",
        "extract_prompt": """
        Decompose this review by:
        1. Extract and list the positive aspects:
        - Praises for visuals, acting, or overall enjoyment
        - Any other positive elements mentioned
        
        2. Extract and list the negative aspects:
        - Criticisms of plot, pacing, or other shortcomings
        - Any other negative elements mentioned
        
        Format your response exactly as:
        Positive Points:
        - [point 1]
        - [point 2]
        ...
        
        Negative Points:
        - [point 1]
        - [point 2]
        ...
        """,
        "classification_prompt": """
        Based on the decomposed positive and negative aspects:
        - Assess whether the overall sentiment leans more toward positive or negative
        - Consider the weight and importance of each point
        
        Only respond with 'positive' or 'negative'.
        """,
        "description": "Two-step chain: aspect decomposition followed by weighted classification",
    },
    "star_rating": {
        "chain_type": "star_rating",
        "rating_prompt": """
        You are an expert movie critic trained to evaluate reviews and assign a **1-to-5 star rating** based on your understanding of how IMDb reviews are written. 
        a review can be positive, negative, or mixed.
        
        Only respond with the following format:
        rating: int (1-5),
        sentiment: "positive" or "negative" or "mixed"
        
        Your task is to:
        1. Analyze the review for key aspects that IMDb reviewers typically emphasize.
        2. Assign a **1-to-5 star rating** based on the review's tone, details, and overall impression.
        3. Use the assigned rating to classify the sentiment:
        - Positive Sentiment: For ratings 4 or 5 stars.
        - Negative Sentiment: For ratings 1, 2,.
        - Mixed Sentiment: For ratings 3 stars.
        Evaluation Criteria:

        Consider the following aspects while interpreting the review:
        1. Story and Plot: Is the story engaging, coherent, and well-executed?
        2. Acting and Characters: Are the performances strong, and are the characters well-developed?
        3. Direction and Production Quality: Does the movie excel in direction, cinematography, and overall production value?
        4. Emotional Impact and Viewer Engagement: Does the movie evoke strong emotions or leave a lasting impression?

        Scoring System:
        1. Read the review carefully, identify the tone, and extract relevant details.
        2. Based on the review, assign a 1-to-5 star rating:
        - 1 Star: The review is highly negative with no redeeming qualities mentioned. Sentiment: negative
        - 2 Stars: The review is mostly negative with minimal positive aspects. Sentiment: negative
        - 3 Stars: The review is mixed but leaning towards negative. Sentiment: mixed
        - 4 Stars: The review is mostly positive with minor issues. Sentiment: positive
        - 5 Stars: The review is overwhelmingly positive with no major flaws. Sentiment: positive

        3. Provide the final sentiment based on the rating.

        Output Format:
        rating: int (1-5),
        sentiment: "positive" or "negative" or "mixed"
        """,
        "resolution_prompt": """
        You are an expert film critic specializing in resolving mixed or ambiguous reviews.
        For reviews that show both positive and negative aspects, carefully weigh the following:

        1. Impact of Criticisms:
        - Are the negative points deal-breakers or minor issues?
        - Do they significantly impact the overall viewing experience?

        2. Strength of Praise:
        - Is the positive feedback substantial or superficial?
        - Do the positive aspects outweigh the negative ones?

        3. Reviewer's Tone:
        - Does the overall tone lean more towards recommendation or warning?
        - What's the final impression the reviewer wants to convey?

        Based on these factors, classify the review as either 'positive', 'negative'
        Only respond with exactly one word: either 'positive', 'negative'
        """,
        "description": "Two-step chain: star rating followed by resolution of mixed ratings",
    },
}


def get_experiment_config(experiment_type: str, experiment_name: str) -> Dict:
    """
    Get configuration for a specific experiment.

    Args:
        experiment_type: Type of experiment ('prompt', 'params', or 'chain')
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

    elif experiment_type == "chain":
        if experiment_name not in CHAIN_EXPERIMENTS:
            raise ValueError(
                f"Unknown chain experiment: {experiment_name}. "
                f"Available options: {list(CHAIN_EXPERIMENTS.keys())}"
            )
        return CHAIN_EXPERIMENTS[experiment_name]

    else:
        raise ValueError(
            f"Unknown experiment type: {experiment_type}. "
            "Available options: 'prompt', 'params', or 'chain'"
        )
