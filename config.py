# config.py
import argparse

# Define global config
config = {}

def parse_args():
    """Parse command-line arguments and store them in the global config."""
    parser = argparse.ArgumentParser(description="XGBoost Model Training")

    # General parameters
    parser.add_argument("--model", choices=["xgboost", "catboost"], default="xgboost", help="Model to use")
    parser.add_argument("--features", choices=["ms", "ms-smiles", "dual"], default="dual", help="Feature set")
    parser.add_argument("--predict", action="store_true", help="Show plots")
    parser.add_argument("--graphs", action="store_true", help="Show plots")
    parser.add_argument("--optimize", action="store_true", help="Run HPO")
    parser.add_argument("--test_per_amine", action="store_true", help="Test performance of model against each amine")

    args = parser.parse_args()
    config.update(vars(args))  # Store parsed arguments in config dict


parse_args()  # Automatically parse when imported
