import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

from amine_solubility import load_data

# Load the data
def select_features():
    df = load_data()
    
    # Keep only amines in water experiments
    df = df[df['In:'] == 'Water']
    
    # Filter the relevant features
    features = [
        'T [K]',
        'C in solute',
        'H in solute',
        'N in solute',
        'O in solute',
        'Molecular weight solute [g/mol]',
        'XLogP3-AA solute',
        'Hydrogen bond donor count solute',
        'Hydrogen bond acceptor count solute',
        'Rotatable bond count solute',
        'Topological polar surface area solute [Å²]',
        'Complexity solute',
        'x' # solubility
    ]

    df = df[features].dropna()

    # Rename columns with brackets to parens to avoid issues with XGBoost
    df.rename(columns=lambda col: col.replace('[', '(').replace(']', ')'), inplace=True)

    return df

def train_model(X_train, y_train):    
    """Train and evaluate an XGBoost model"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100,
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def plot_predictions(model, X_test, y_test):
    # Calculate rmse
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test['T (K)'], y_test, label='True', alpha=0.6, edgecolor='k', s=40)
    plt.scatter(X_test['T (K)'], y_pred, label='Predicted', alpha=0.6, edgecolor='k', s=40)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Solubility', fontsize=12)
    plt.title('True vs Predicted Solubility', fontsize=14)
    plt.legend(fontsize=10)

    # Add RMSE as a text box
    plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", fontsize=12, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
    plt.show()

def plot_parity(model, X_test, y_test):
    """Show a parity plot of true vs predicted solubility"""
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=40, label='Data Points')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

    # Labels and title
    plt.xlabel('True Solubility', fontsize=12)
    plt.ylabel('Predicted Solubility', fontsize=12)
    plt.title('Parity Plot: True vs Predicted Solubility', fontsize=14)

    # Add RMSE as a text box
    plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", fontsize=12, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Visualize feature importance
def plot_feature_importance(model, feature_names):
    xgb.plot_importance(model, importance_type='weight', show_values=False)
    plt.xticks(rotation=45)
    plt.show()


def main():
    data = select_features()
    X = data.drop(columns=['x'])
    y = data['x']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    plot_predictions(model, X_test, y_test)
    plot_parity(model, X_test, y_test)
    plot_feature_importance(model, X.columns)
    
    return model

if __name__ == "__main__":
    main()
