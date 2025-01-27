"""
Main experiment runner for model comparisons.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List

from data.data_loader import load_dataset_subset
from experiments.experiment_configs import get_experiment_config
from experiments.plot_metrics import create_experiment_visualizations
from src.config import logger
from src.evaluation import PredictionResult, evaluate_model_performance
from src.models import load_model
from experiments.chain import (
    summary_chain, 
    confidence_chain, 
    decomposition_chain,
    star_rating_chain
)


def validate_prediction(prediction: str) -> str:
    """
    Validate and extract sentiment from model prediction.

    Args:
        prediction: Raw model prediction - either a single word ('positive'/'negative') 
                   or structured format
    Returns:
        Validated sentiment ('positive' or 'negative')

    Raises:
        ValueError: If prediction is not a valid sentiment
    """
    try:
        # Clean up prediction
        prediction = prediction.strip().lower()
        
        # If it's a single word response
        if prediction in {"positive", "negative"}:
            return prediction
            
        # If it's structured format, parse it
        lines = [line.strip() for line in prediction.split('\n') if line.strip()]
        
        # Check first and last lines for sentiment
        lines_to_check = [lines[0]]
        if len(lines) > 1:
            lines_to_check.append(lines[-1])
            
        for line in lines_to_check:
            if "sentiment:" in line.lower():
                sentiment = line.split("sentiment:")[1].strip().strip('"').strip('*')
                if sentiment in {"positive", "negative"}:
                    return sentiment
                
        raise ValueError(
            f"Invalid sentiment: '{prediction}'. "
            "Must be exactly 'positive' or 'negative'"
        )

    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid prediction format: {str(e)}")


def run_model_evaluation(
    model_size: str, test_cases: List[Dict], system_prompt: str, inference_params: Dict
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
            # Check if using chain prompts
            if isinstance(system_prompt, dict) and "chain_type" in system_prompt:
                if system_prompt["chain_type"] == "summary":
                    raw_prediction = summary_chain(
                        model,
                        case["input"],
                        system_prompt["summary_prompt"],
                        system_prompt["classification_prompt"]
                    )
                elif system_prompt["chain_type"] == "confidence":
                    raw_prediction = confidence_chain(
                        model,
                        case["input"],
                        system_prompt["student_prompt"],
                        system_prompt["teacher_prompt"]
                    )
                elif system_prompt["chain_type"] == "decomposition":
                    raw_prediction = decomposition_chain(
                        model,
                        case["input"],
                        system_prompt["extract_prompt"],
                        system_prompt["classification_prompt"]
                    )
                elif system_prompt["chain_type"] == "star_rating":
                    raw_prediction = star_rating_chain(
                        model,
                        case["input"],
                        system_prompt["rating_prompt"],
                        system_prompt["resolution_prompt"]
                    )
            else:
                response = model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": case["input"]},
                    ],
                    **{k: v for k, v in inference_params.items() if k != "description"},
                )
                raw_prediction = response["choices"][0]["message"]["content"]

            inference_time = time() - start_time

            try:
                prediction = validate_prediction(raw_prediction)
                prediction_results.append(
                    PredictionResult(
                        input_text=case["input"],
                        true_label=case["label"],
                        predicted_label=prediction,
                        response_time=inference_time,
                    )
                )

            except ValueError as e:
                invalid_predictions.append(
                    {
                        "input": case["input"],
                        "raw_prediction": raw_prediction,
                        "error": str(e),
                    }
                )
                logger.warning(f"Sample {i}: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing case {i}: {str(e)}")
            invalid_predictions.append(
                {"input": case["input"], "raw_prediction": "ERROR", "error": str(e)}
            )

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
        "invalid_examples": invalid_predictions[
            :10
        ],  # Store first 10 invalid predictions as examples
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

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                **converted_info,
                "results": converted_results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


def run_experiment(experiment_type: str, experiment_name: str, sample_size: int):
    """
    Run a specific experiment configuration.

    Args:
        experiment_type: Type of experiment ('prompt', 'params', or 'chain')
        experiment_name: Name of the specific experiment configuration
        sample_size: Number of samples to use
    """
    # Get experiment configuration
    config = get_experiment_config(experiment_type, experiment_name)

    # Setup output directory
    base_dir = Path("experiment_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    inference_params = get_experiment_config("params", "default")
    
    for model_size in ["0.5B", "1.5B"]:
        if experiment_type == "chain":
            # For chain experiments, we pass the chain config directly
            results[model_size] = run_model_evaluation(
                model_size,
                test_cases,
                config,  # This contains the chain configuration
                inference_params,
            )
        elif experiment_type == "prompt":
            results[model_size] = run_model_evaluation(
                model_size,
                test_cases,
                config["system"],
                inference_params,
            )
        else:  # params experiment
            results[model_size] = run_model_evaluation(
                model_size,
                test_cases,
                get_experiment_config("prompt", "CoT_few_shot")["system"],
                config,
            )

    # Save results and create visualizations
    experiment_info = {
        "experiment_name": f"{experiment_type}: {experiment_name}",
        "experiment_type": experiment_type,
        "configuration": config,
        "sample_size": sample_size,
    }

    save_experiment_results(results, experiment_info, output_dir)
    create_experiment_visualizations(output_dir)

    logger.info(f"\nExperiment completed. Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis experiments")
    parser.add_argument(
        "--type",
        choices=["prompt", "params", "chain"],  # Add 'chain' as an option
        required=True,
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--name", required=True, help="Name of the experiment configuration"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to use (default: 100)",
    )

    args = parser.parse_args()
    run_experiment(args.type, args.name, args.sample_size)


if __name__ == "__main__":
    main()
