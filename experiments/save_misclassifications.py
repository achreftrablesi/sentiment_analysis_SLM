"""
Script to run inference on the dataset and save misclassified examples to CSV.
"""

import csv
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List

from src.config import logger
from data.data_loader import load_dataset_subset
from src.models import load_model
from experiments.experiment_configs import get_experiment_config


def run_sentiment_analysis(
    review: str,
    model,
    config: Dict,
) -> Dict:
    """
    Run sentiment analysis on a single review.

    Args:
        review: The review text to analyze
        model: The loaded model instance
        config: The experiment configuration containing the system prompt

    Returns:
        Dict containing prediction and timing information
    """
    start_time = time()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": f"<review>{review}</review>"},
        ],
        temperature=0.0  # Using a default temperature, could be made configurable
    )
    inference_time = time() - start_time
    
    prediction = response["choices"][0]["message"]["content"].strip().lower()
    
    return {
        "prediction": prediction,
        "response_time": inference_time
    }


def save_misclassifications(
    dataset_size: int = 100,
    model_size: str = "0.5B",
    experiment_name: str = "zero_shot",
    output_dir: str = "misclassification_analysis"
) -> None:
    """
    Analyze the dataset and save misclassified examples to CSV.

    Args:
        dataset_size: Number of samples to analyze
        model_size: Size of the model to use ('0.5B' or '1.5B')
        experiment_name: Name of the experiment configuration to use
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get experiment configuration
    try:
        config = get_experiment_config("prompt", experiment_name)
        logger.info(f"Using experiment configuration: {experiment_name}")
        logger.info(f"Description: {config['description']}")
    except ValueError as e:
        logger.error(f"Error loading experiment configuration: {e}")
        return
    
    # Load model and dataset
    logger.info(f"Loading {model_size} model...")
    model = load_model(model_size)
    
    logger.info(f"Loading dataset (size: {dataset_size})...")
    dataset = load_dataset_subset(size=dataset_size)
    
    # Prepare CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"misclassifications_{model_size}_{experiment_name}_{timestamp}.csv"
    
    misclassified = []
    total_time = 0
    correct_predictions = 0
    
    logger.info("Starting analysis...")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Review Text',
            'True Label',
            'Predicted Label',
            'Response Time (s)',
            'Experiment Type'
        ])
        
        for i, item in enumerate(dataset, 1):
            if i % 10 == 0:
                logger.info(f"Processing review {i}/{dataset_size}")
            
            true_label = "positive" if item["label"] == 0 else "negative"
            
            try:
                result = run_sentiment_analysis(
                    review=item["review"],
                    model=model,
                    config=config
                )
                
                total_time += result["response_time"]
                
                if result["prediction"] != true_label:
                    misclassified.append({
                        'review': item["review"],
                        'true_label': true_label,
                        'predicted': result["prediction"],
                        'time': result["response_time"]
                    })
                    # Write to CSV immediately
                    writer.writerow([
                        item["review"],
                        true_label,
                        result["prediction"],
                        f"{result['response_time']:.3f}",
                        experiment_name
                    ])
                else:
                    correct_predictions += 1
                    
            except Exception as e:
                logger.error(f"Error processing review {i}: {str(e)}")
    
    # Calculate and log statistics
    accuracy = correct_predictions / dataset_size
    avg_time = total_time / dataset_size
    
    logger.info("\nAnalysis Complete!")
    logger.info(f"Experiment type: {experiment_name}")
    logger.info(f"Total samples processed: {dataset_size}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Misclassifications: {len(misclassified)}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Average processing time: {avg_time:.3f} seconds")
    logger.info(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and save model misclassifications')
    parser.add_argument('--size', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--model', type=str, default="0.5B", choices=['0.5B', '1.5B'],
                      help='Model size to use')
    parser.add_argument('--experiment', type=str, default="zero_shot",
                      choices=['zero_shot', 'one_shot', 'few_shot', 'CoT', 'CoT_few_shot'],
                      help='Experiment configuration to use')
    parser.add_argument('--output', type=str, default="misclassification_analysis",
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    save_misclassifications(
        dataset_size=args.size,
        model_size=args.model,
        experiment_name=args.experiment,
        output_dir=args.output
    ) 