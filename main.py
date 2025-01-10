"""
Simple Streamlit app for sentiment analysis.
"""
import time

import streamlit as st

from src.config import (
    CLASSIFIER_PROMPT_0_5B,
    CLASSIFIER_PROMPT_1_5B,
    MAX_TOKENS,
    MODEL_MAPPING,
    TEMPERATURE,
    USER_PROMPT,
    logger,
)
from src.models import load_model


def analyze_sentiment(model_size: str, text: str) -> str:
    """Run sentiment analysis on a single review."""
    # Select the appropriate prompt based on model size
    classifier_prompt = (
        CLASSIFIER_PROMPT_1_5B
        if MODEL_MAPPING[model_size] == "1.5B"
        else CLASSIFIER_PROMPT_0_5B
    )
    logger.info(f"Using model: {MODEL_MAPPING[model_size]}")

    # Load model and run prediction
    model = load_model(MODEL_MAPPING[model_size])
    start_time = time.time()
    try:
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": classifier_prompt},
                {"role": "user", "content": USER_PROMPT.format(review=text)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return "Error analyzing sentiment"


def main():
    st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="wide")

    # Custom CSS to improve appearance
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            font-size: 1.1rem;
            line-height: 1.5;
        }
        .sentiment-result {
            font-size: 2.5rem;
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            text-align: center;
            margin-top: 20px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("Movie Review Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Model selection with descriptions
        model_size = st.selectbox(
            "Select Model",
            options=list(MODEL_MAPPING.keys()),
            help="Choose between a faster but simpler model or a more capable but slower one",
        )

        # Review input
        review = st.text_area(
            "Enter movie review:",
            height=300,
            placeholder="Type or paste your movie review here...",
        )

    with col2:
        st.markdown("### Sentiment Analysis Result")
        if review:
            try:
                with st.spinner("Analyzing sentiment..."):
                    sentiment = analyze_sentiment(model_size, review)
                st.markdown(
                    f"""
                    <div class="sentiment-result">
                        {sentiment.upper()}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error analyzing sentiment")
        else:
            st.info("Enter a review on the left to see the sentiment analysis result.")


if __name__ == "__main__":
    main()
