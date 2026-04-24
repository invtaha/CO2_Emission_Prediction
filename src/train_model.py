"""
Train a simple linear regression model to predict CO2 emissions
based on engine size. Saves the model and the regression plot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "CO2_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "simple_linear_model.pkl")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PLOT_PATH = os.path.join(IMAGES_DIR, "co2_regression_plot.png")

def load_data(path):
    """Load CO2 dataset and return features (X) and target (y)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    x = df[["ENGINESIZE"]]
    y = df["CO2EMISSIONS"]
    return x, y

def main():
    print("Starting model training...")

    #data load
    x, y = load_data(DATA_PATH)
    print(f"Dataset loaded. Shape: {x.shape[0]} samples")

    #data split
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}\nTest: {len(X_test)}")

    #model training
    model = LinearRegression()
    model.fit(X_train, y_train)


    #prediction and evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"   MAE  : {mae:.2f} g/km")
    print(f"   MSE  : {mse:.2f}")
    print(f"   R²   : {r2:.4f}")

    #save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    #draw the plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='steelblue', alpha=0.6, label='Actual values')
    plt.plot(X_test, y_pred, color='firebrick', linewidth=2, label='Regression line')
    plt.xlabel('Engine Size (L)', fontsize=12)
    plt.ylabel('CO₂ Emissions (g/km)', fontsize=12)
    plt.title('CO₂ Prediction using Simple Linear Regression', fontsize=14, weight='bold')
    plt.legend()

    #inserting the R2 value on the graph
    plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {PLOT_PATH}")


    #show model details
    theta0 = model.intercept_
    theta1 = model.coef_[0]
    print(f"\nModel Equation: CO2 = {theta0:.2f} + ({theta1:.2f}) × EngineSize")


if __name__ == "__main__":
    main()