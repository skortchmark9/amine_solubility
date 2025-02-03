import pprint
import pandas as pd
from sklearn.base import is_classifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn_compat.utils import get_tags

from amine_solubility import load_data

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


selected_features = [
    'T [K]',
    # 'C in solute',
    # 'H in solute',
    # 'N in solute',
    'O in solute',
    'XLogP3-AA solute',
    'Hydrogen bond donor count solute',
    'Hydrogen bond acceptor count solute',
    'Rotatable bond count solute',
    'Topological polar surface area solute [Å²]',
    # 'Complexity solute',
    'Undefined atom stereocenter count solute',
]

target = ['x']


def pearson_correlation_coefficient():
    df = load_data(text=False)
    df = df[selected_features].dropna()
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Feature Correlation Matrix")
    plt.show()


def vif():
    df = load_data(text=False)
    df = df[selected_features].dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

    print(vif_data)


def select_features():
    print("Selecting features...")
    df = load_data()
    
    # Keep only amines in water experiments
    df = df[df['In:'] == 'Water']
    
    # Filter the relevant features


    df = df[selected_features + target].dropna()

    # Rename columns with brackets to parens to avoid issues with XGBoost
    df.rename(columns=lambda col: col.replace('[', '(').replace(']', ')'), inplace=True)

    return df

def train_model_simple(X_train, y_train):    
    """Train and evaluate an XGBoost model"""

    # From testing 2/3/2021
    optimized_hyperparameters = {
        "random_state": 42,
        "colsample_bytree": 0.8,
        "learning_rate": 0.01,
        "max_depth": 4,
        "n_estimators": 400,
        "subsample": 0.9
    }
    model = xgb.XGBRegressor(
        **optimized_hyperparameters
    )
    
    model.fit(X_train, y_train)
    return model


def train_model_optimized():
    data = select_features()
    print("Optimizing hyperparameters...")
    X = data.drop(columns=['x'])
    y = data['x']

    param_grid = {
        "random_state": [42],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8]
    }

    xgb_model = xgb.XGBRegressor(**param_grid)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=10,
        n_jobs=-1
    )

    grid_search.fit(X.values, y.values)

    # Get best model
    best_model = grid_search.best_estimator_

    print(f"Best parameters with score: {grid_search.best_score_:4f}")
    print("\n".join(
        [f"\t{k}: {v}" for k, v in grid_search.best_params_.items()]
    ))
    return best_model


def plot_predictions(model, X_test, y_test):
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

def plot_feature_importance(model):
    xgb.plot_importance(model, importance_type='weight', show_values=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse: .4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")



def main(optimize=True):
    data = select_features()
    X = data.drop(columns=['x'])
    y = data['x']
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if optimize:
        model = train_model_optimized()
    else:
        model = train_model_simple(X_train, y_train)

    feature_names = list(X.columns)
    model.get_booster().feature_names = feature_names

    pearson_correlation_coefficient()
    
    print_metrics(model, X_test, y_test)
    plot_predictions(model, X_test, y_test)
    plot_parity(model, X_test, y_test)
    plot_feature_importance(model)
    
    return model

if __name__ == "__main__":
    main()
