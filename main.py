"""
Simple Streamlit app for sentiment analysis.
"""
import streamlit as st
from src.models import load_model
from src.config import TEMPERATURE, MAX_TOKENS, CLASSIFIER_PROMPT, USER_PROMPT

MODEL_MAPPING = {
    "Fast & Compact (0.5B)": "0.5B",
    "Strong & Capable (1.5B)": "1.5B"
}

def analyze_sentiment(model_size: str, text: str) -> str:
    """Run sentiment analysis on a single review."""
    
    
    # Load model and run prediction
    model = load_model(MODEL_MAPPING[model_size])
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(review=text)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    return response["choices"][0]["message"]["content"]

def main():
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        layout="wide"
    )
    
    # Custom CSS to improve appearance
    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.title("Movie Review Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Model selection with descriptions
        model_size = st.selectbox(
            "Select Model",
            options=list(MODEL_MAPPING.keys()),
            help="Choose between a faster but simpler model or a more capable but slower one"
        )
        
        # Review input
        review = st.text_area(
            "Enter movie review:",
            height=300,
            placeholder="Type or paste your movie review here..."
        )

    with col2:
        st.markdown("### Sentiment Analysis Result")
        if review:
            try:
                with st.spinner("Analyzing sentiment..."):
                    sentiment = analyze_sentiment(model_size, review)
                st.markdown(f"""
                    <div class="sentiment-result">
                        {sentiment.upper()}
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
        else:
            st.info("Enter a review on the left to see the sentiment analysis result.")

if __name__ == "__main__":
    main()
