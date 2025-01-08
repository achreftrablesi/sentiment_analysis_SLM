"""
Main experiment runner for model comparisons.
"""
from typing import List, Dict
import json
from pathlib import Path
from datetime import datetime
from time import time
import argparse

from src.models import load_model
from src.evaluation import PredictionResult, evaluate_model_performance
from data.data_loader import load_dataset_subset
from experiments.plot_metrics import create_experiment_visualizations
from experiments.experiment_configs import get_experiment_config
from src.config import logger


def validate_prediction(prediction: str) -> str:
    """
    Validate that the model prediction is exactly 'positive' or 'negative'.

    Args:
        prediction: Raw model prediction

    Returns:
        Validated prediction ('positive' or 'negative')

    Raises:
        ValueError: If prediction is not exactly 'positive' or 'negative'
    """
    prediction = prediction.strip().lower()
    
    if prediction not in {'positive', 'negative'}:
        raise ValueError(
            f"Invalid prediction: '{prediction}'. "
            "Prediction must be exactly 'positive' or 'negative'"
        )
    
    return prediction


def run_model_evaluation(
    model_size: str,
    test_cases: List[Dict],
    system_prompt: str,
    inference_params: Dict
) -> Dict:
    """Run evaluation for a single model configuration."""
    logger.info(f"\nEvaluating {model_size} model...")
    model = load_model(model_size)
    prediction_results = []
    invalid_predictions = []

    for i, case in enumerate(test_cases, 1):
        if i % 10 == 0:
            logger.info(f"Processing sample {i}/{len(test_cases)}")

        start_time = time()
        try:
            response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": case["input"]},
                ],
                **{k: v for k, v in inference_params.items() if k != "description"}
            )
            inference_time = time() - start_time

            raw_prediction = response["choices"][0]["message"]["content"]
            
            try:
                prediction = validate_prediction(raw_prediction)
                prediction_results.append(
                    PredictionResult(
                        input_text=case["input"],
                        true_label=case["label"],
                        predicted_label=prediction,
                        response_time=inference_time
                    )
                )
            
            except ValueError as e:
                invalid_predictions.append({
                    "input": case["input"],
                    "raw_prediction": raw_prediction,
                    "error": str(e)
                })
                logger.warning(f"Sample {i}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing case {i}: {str(e)}")
            invalid_predictions.append({
                "input": case["input"],
                "raw_prediction": "ERROR",
                "error": str(e)
            })

    if not prediction_results:
        raise ValueError("No valid predictions were generated")

    metrics = evaluate_model_performance(prediction_results)
    
    # Add invalid prediction information
    total_cases = len(test_cases)
    valid_cases = len(prediction_results)
    invalid_cases = len(invalid_predictions)
    
    metrics["prediction_coverage"] = {
        "total_samples": total_cases,
        "valid_predictions": valid_cases,
        "invalid_predictions": invalid_cases,
        "coverage_percentage": (valid_cases / total_cases) * 100,
        "invalid_percentage": (invalid_cases / total_cases) * 100,
        "invalid_examples": invalid_predictions[:10]  # Store first 10 invalid predictions as examples
    }

    return metrics


def save_experiment_results(results: Dict, experiment_info: Dict, output_dir: Path):
    """Save experiment results and configuration."""
    def convert_to_native_types(obj):
        """Convert numpy types to native Python types."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        return obj

    # Convert results to native Python types
    converted_results = convert_to_native_types(results)
    converted_info = convert_to_native_types(experiment_info)

    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            **converted_info,
            "results": converted_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)


def run_experiment(
    experiment_type: str,
    experiment_name: str,
    sample_size: int
):
    """
    Run a specific experiment configuration.
    
    Args:
        experiment_type: Type of experiment ('prompt' or 'params')
        experiment_name: Name of the specific experiment configuration
        sample_size: Number of samples to use
    """
    # Get experiment configuration
    config = get_experiment_config(experiment_type, experiment_name)
    
    # Setup output directory
    base_dir = Path("experiment_results")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / f"{experiment_type}_{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"\nLoading dataset (size: {sample_size})...")
    dataset = load_dataset_subset(size=sample_size)
    test_cases = [
        {
            "input": item["review"],
            "label": "positive" if item["label"] == 0 else "negative",
        }
        for item in dataset
    ]

    # Run experiment
    results = {}
    for model_size in ["0.5B", "1.5B"]:
        if experiment_type == "prompt":
            results[model_size] = run_model_evaluation(
                model_size,
                test_cases,
                config["system"],
                get_experiment_config("params", "default")
            )
        else:  # params experiment
            results[model_size] = run_model_evaluation(
                model_size,
                test_cases,
                get_experiment_config("prompt", "basic")["system"],
                config
            )

    # Save results and create visualizations
    experiment_info = {
        "experiment_name": f"{experiment_type}: {experiment_name}",
        "experiment_type": experiment_type,
        "configuration": config,
        "sample_size": sample_size
    }
    
    save_experiment_results(results, experiment_info, output_dir)
    create_experiment_visualizations(output_dir)
    
    logger.info(f"\nExperiment completed. Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis experiments")
    parser.add_argument(
        "--type",
        choices=["prompt", "params"],
        required=True,
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the experiment configuration"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of samples to use (default: 50)"
    )

    args = parser.parse_args()
    run_experiment(args.type, args.name, args.sample_size)


if __name__ == "__main__":
    main()