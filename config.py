# config.py
import argparse

# Define global config
config = {}

def parse_args():
    """Parse command-line arguments and store them in the global config."""
    parser = argparse.ArgumentParser(description="XGBoost Model Training")

    # General parameters
    parser.add_argument("--model", choices=["ms", "dual"], default="dual", help="Model to train")
    parser.add_argument("--predict", action="store_true", help="Show plots")
    parser.add_argument("--graphs", action="store_true", help="Show plots")
    parser.add_argument("--optimize", action="store_true", help="Run HPO")
    parser.add_argument("--smiles", action="store_true", help="Include smiles features")

    args = parser.parse_args()
    config.update(vars(args))  # Store parsed arguments in config dict


parse_args()  # Automatically parse when imported
