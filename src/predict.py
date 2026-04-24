"""
Load a trained linear regression model and predict CO2 emissions
for a given engine size (command-line argument or interactive input).
"""

import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "simple_linear_model.pkl")

def load_model(path):
    """Load the trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    return joblib.load(path)

def main():
    model = load_model(MODEL_PATH)

    #getting engine size from command line argument
    if len(sys.argv) > 1:
        try:
            engine_size = float(sys.argv[1])
        except ValueError:
            print("Error: Engine size must be a number.")
            sys.exit(1)
    else:
        try:
            engine_size = float(input("Enter engine size (L): "))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            sys.exit(1)

    #prediction
    co2_pred = model.predict([[engine_size]])[0]


    print(f"\nPredicted CO₂ emission for {engine_size}L engine: {co2_pred:.2f} g/km")


if __name__ == "__main__":
    main()
