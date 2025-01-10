# Movie Review Sentiment Analysis

This repository contains a project for analyzing movie reviews' sentiment using small language models (SLMs). The goal is to compare the performance of two quantized SLMs from the Qwen2.5 family: a 0.5B parameter model and a 1.5B parameter model. The project explores prompt engineering, inference parameter tuning, and evaluation metrics to maximize the accuracy and efficiency of sentiment analysis.

---

## Project Structure

```
.
├── data                  # Dataset-related scripts and files
│   └── data_loader.py    # Script to load and preprocess dataset subsets
├── experiment_results/   # Directory where experiment outputs (plots, results) are stored
├── experiments           # Experimentation scripts and configurations
│   ├── experiment_configs.py # Definitions for various experiment configurations
│   ├── plot_metrics.py       # Utilities for generating visualizations of experiment results
│   └── run_experiments.py    # Script to automate experiments with different prompts and parameters
├── src                  # Core project files
│   ├── config.py         # Configuration file for prompts, model settings, and logging
│   ├── evaluation.py     # Evaluation functions for performance metrics and timing analysis
│   ├── inference.py      # Script to run inference using predefined models and test messages
│   └── models.py         # Contains model loading and initialization functions
├── .gitignore            # Git ignore file
├── main.py               # Streamlit app for interactive sentiment analysis
├── requirements.txt      # Python dependencies for the project
├── run.sh                # Shell script for setting up and running the project
```

### File Descriptions

- **`data/data_loader.py`**: Loads and preprocesses a subset of the dataset for experiments.
- **`experiment_results/`**: Directory for storing results, including plots and logs, from experimental runs.
- **`experiments/experiment_configs.py`**: Holds configuration data for prompt engineering and inference parameter experiments.
- **`experiments/plot_metrics.py`**: Generates visual comparisons (e.g., accuracy, timing) between models based on experimental results.
- **`experiments/run_experiments.py`**: Automates running experiments with different prompts and inference settings.
- **`src/config.py`**: Stores configuration settings like prompts, model repositories, and inference parameters.
- **`src/evaluation.py`**: Provides methods to compute metrics such as accuracy, confusion matrix, and response time.
- **`src/inference.py`**: Implements inference logic for the sentiment analysis task. It processes input reviews using the selected model and returns predictions.
- **`src/models.py`**: Contains functions to load and initialize the Qwen2.5 models with specified context windows and parameters.
- **`main.py`**: A Streamlit-based web interface for users to interact with the models and analyze review sentiments.
- **`requirements.txt`**: Lists the Python libraries required to run the project.
- **`run.sh`**: A setup and execution script for initializing the environment and launching the app.

---

## Setup Instructions

### Compatibility

- **Python Version**: 3.9.9
- **OS**: Linux or Windows (adjust file permissions for `run.sh` on Linux)

### Option 1: Using `run.sh`

1. Clone this repository:
   ```bash
   git clone https://github.com/achreftrablesi/sentiment_analysis_SLM.git
   cd sentiment_analysis_SLM
   ```
2. Ensure the script is executable on Linux:
   ```bash
   chmod +x run.sh
   ```
3. Run the setup script:
   ```bash
   ./run.sh
   ```
   This will:
   - Create a virtual environment.
   - Install required dependencies.
   - Initialize the models (takes time for the first run).
   - Start the Streamlit application for sentiment analysis.

### Option 2: Manual Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/achreftrablesi/sentiment_analysis_SLM.git
   cd sentiment_analysis_SLM
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Initialize the models:
   ```bash
   python -m src.models --size 0.5B
   python -m src.models --size 1.5B
   ```
   **Note**: Loading the models will take time during the first run, but subsequent runs will be faster.
   
5. Run the Streamlit app: ( for future runs, you only need to run this command)
   ```bash
   streamlit run main.py
   ```

---

## Usage

### Streamlit Interface
Once the Streamlit app is running, you can:
- Select a model (`0.5B` or `1.5B`).
- Enter a movie review in the provided text area.
- View the sentiment analysis result.

### Running Experiments
To run experiments comparing models or configurations, use:
```bash
python experiments/run_experiments.py --type <experiment-type> --name <experiment-name> --sample-size <sample-size>
```
- `experiment-type`: `prompt` or `params`.
- `experiment-name`: Name of the specific experiment configuration (e.g., `CoT_few_shot`).
- `sample-size`: Number of samples to use (default: 100).

Results and plots will be saved in the `experiment_results/` directory.

---

## Example Results

Here is a template of what the `results.json` file will look like after an experiment:

```json
{
  "experiment_name": "prompt: CoT_few_shot",
  "experiment_type": "prompt",
  "configuration": {
    "system": "You are a movie review classifier...",
    "description": "Chain of Thought with Few-Shot examples showing the reasoning process"
  },
  "sample_size": 400,
  "results": {
    "0.5B": {
      "accuracy": 0.94,
      "true_positives": 47,
      "true_negatives": 47,
      "false_positives": 3,
      "false_negatives": 3,
      "timing": {
        "average_response_time": 1.131,
        "std_response_time": 0.827,
        "max_response_time": 4.386,
        "min_response_time": 0.237,
        "total_inference_time": 113.093
      },
      "prediction_coverage": {
        "total_samples": 400,
        "valid_predictions": 400,
        "invalid_predictions": 0,
        "coverage_percentage": 100.0,
        "invalid_percentage": 0.0,
        "invalid_examples": []
      }
    },
    "1.5B": {
      "accuracy": 0.92,
      "true_positives": 43,
      "true_negatives": 49,
      "false_positives": 1,
      "false_negatives": 7,
      "timing": {
        "average_response_time": 3.155,
        "std_response_time": 2.315,
        "max_response_time": 12.404,
        "min_response_time": 0.655,
        "total_inference_time": 315.489
      },
      "prediction_coverage": {
        "total_samples": 400,
        "valid_predictions": 400,
        "invalid_predictions": 0,
        "coverage_percentage": 100.0,
        "invalid_percentage": 0.0,
        "invalid_examples": []
      }
    }
  },
  "timestamp": "2025-01-09T12:24:51.359699"
}
```

---

For questions or further assistance, please contact the repository maintainer.


## Full Report

For a comprehensive analysis, including insights, challenges, and detailed visualizations, please refer to the [full Report](https://docs.google.com/document/d/1An8ad5i0OgzOa-TYQeTW994Li8uH01QFndAKL6RWJXY/edit?usp=sharing).