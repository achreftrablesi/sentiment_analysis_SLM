"""
Data loader for the sentiment analysis project.

"""

from datasets import concatenate_datasets, load_dataset

from src.config import DATASET_PATH, logger


def load_dataset_subset(size: int = 1000):
    """
    Load the IMDB movie reviews dataset and select a balanced subset.

    Returns:
        A subset of the Hugging Face IMDB movie reviews dataset containing reviews and labels.
    """
    size_per_label = size // 2
    # Load the dataset
    try:
        dataset = load_dataset(DATASET_PATH)
    except Exception as e:
        logger.error(f"Error loading the dataset: {e}")
        raise
    reviews = dataset["train"]

    # Separate positive and negative reviews
    positive_reviews = reviews.filter(lambda x: x["label"] == 1)
    negative_reviews = reviews.filter(lambda x: x["label"] == 0)

    # Select a balanced subset of positive and negative reviews
    pos_subset = positive_reviews.shuffle(seed=42).select(range(size_per_label))
    neg_subset = negative_reviews.shuffle(seed=42).select(range(size_per_label))

    balanced_subset = concatenate_datasets([pos_subset, neg_subset]).shuffle(seed=42)

    return balanced_subset


if __name__ == "__main__":
    dataset = load_dataset_subset(size=1000)

    logger.info("Dataset Overview:")
    logger.info(f"Number of reviews: {len(dataset)}")
    logger.info(
        f"Number of positive reviews: {len(dataset.filter(lambda x: x['label'] == 1))}"
    )
    logger.info(
        f"Number of negative reviews: {len(dataset.filter(lambda x: x['label'] == 0))}"
    )
    logger.info("\nExample Reviews:")

    # Print first 4 reviews with their labels
    for i in range(3):
        review = dataset[i]
        sentiment = "Positive" if review["label"] == 1 else "Negative"
        logger.info(f"\nReview #{i+1} ({sentiment}):")
        # Print the review
        logger.info(f"{review['review']}")
