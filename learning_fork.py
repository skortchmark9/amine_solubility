import random
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt

from amine_solubility import load_data
import plotly

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


selected_features_dual = [
    'Solubility of:',
    'In:',
    'T (K)',
    # 'C in solute',
    # 'H in solute',
    # 'N in solute',
    # 'O in solute',
    'XLogP3-AA solute',
    'Hydrogen bond donor count solute',
    'Hydrogen bond acceptor count solute',
    'Rotatable bond count solute',
    'Topological polar surface area solute (Å²)',
    'Complexity solute',
    'Undefined atom stereocenter count solute',
    # 'Molecular weight solute (g/mol)',

    'XLogP3-AA solvent',
    'Hydrogen bond donor count solvent',
    'Hydrogen bond acceptor count solvent',
    'Rotatable bond count solvent',
    'Topological polar surface area solvent (Å²)',
    'Complexity solvent',
    'Undefined atom stereocenter count solvent',
]

target = ['x']

SELECTED_FEATURES = selected_features_dual


def pearson_correlation_coefficient(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Feature Correlation Matrix")
    plt.show()


def vif(df):
    df = df[SELECTED_FEATURES].dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

    print(vif_data)


def select_features(df):
    print("Selecting features...")
    # Keep only amines in water experiments
    # df = df[df['Solubility of:'] == 'Water']

    print("Data size:", df.shape)

    df = df[SELECTED_FEATURES + target].dropna()
    return df

def train_model_simple(X_train, y_train):    
    """Train and evaluate an XGBoost model"""

    # From testing 2/3/2021
    optimized_hyperparameters = {
        "random_state": 42,
        "colsample_bytree": 0.8,
        "learning_rate": 0.01,
        "max_depth": 3,
        "n_estimators": 400,
        "subsample": 0.8,
    }
    # From stefano
    optimized_hyperparameters = {
        'learning_rate': 0.01,
        'n_estimators': 400,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    model = xgb.XGBRegressor(
        **optimized_hyperparameters
    )
    
    model.fit(X_train, y_train)
    return model


def train_model_optimized(X_train, y_train):
    print("Optimizing hyperparameters...")

    param_grid = {
        "random_state": [42],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        # 'objective': 'reg:quantileerror',
        # 'alpha': 0.5,
    }

    xgb_model = xgb.XGBRegressor(**param_grid)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=10,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

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

def build_model(data, optimize=False, graphs=False):
    X = data.drop(columns=['x'])
    y = data['x']
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if optimize:
        model = train_model_optimized(X_train, y_train)
    else:
        model = train_model_simple(X_train, y_train)

    feature_names = list(X.columns)
    model.get_booster().feature_names = feature_names

    
    print_metrics(model, X_test, y_test)
    if graphs:
        pearson_correlation_coefficient(data)
        plot_predictions(model, X_test, y_test)
        plot_parity(model, X_test, y_test)
        plot_feature_importance(model)
    return model

def build_combined_model(df, optimize=False, graphs=True):
    water_in_amine = df[df['Solubility of:'] == 'Water']
    amine_in_water = df[df['Solubility of:'] != 'Water']

    print('Building model for water in amine')
    m_wia = CombinedModel.build(water_in_amine, optimize)

    print('Building model for amine in water')
    m_aiw = CombinedModel.build(amine_in_water, optimize)

    model = CombinedModel(m_wia, m_aiw)
    return model


def predict_one(df, name):
    # partition the df into two parts depending on a condition
    if 'name' in df.keys():
        cond = (df['name'] == name)
    else:
        cond = (df['Solubility of:'] == name) | (df['In:'] == name)
    name_matches = df[cond]
    name_not_matches = df[~cond]

    df_test = name_matches
    df_train = name_not_matches

    model = build_combined_model(df_train, graphs=False, optimize=True)
    y_pred = model.predict(df_test.drop(columns=['x']))
    y_actual = df_test['x']

    # compare the prediction and actual values w/r2
    r2 = r2_score(y_actual, y_pred)



    # Plot the prediction and actual against the temperature
    # using plotly
    fig = plotly.graph_objs.Figure()
    trace_pred = plotly.graph_objs.Scatter(
        x=y_pred,
        y=df_test['T (K)'],
        mode='markers',
        name='Predicted',
        marker=dict(
            color='blue'
        )
    )
    trace_actual = plotly.graph_objs.Scatter(
        x=y_actual,
        y=df_test['T (K)'],
        mode='markers',
        name='Actual',
        marker=dict(
            color='red'
        )
    )
    fig.layout.title = name + (' (R2: %.2f)' % r2)
    fig.update_xaxes(range=[0, 1])
    fig.add_trace(trace_pred)
    fig.add_trace(trace_actual)
    fig.show()

def predict_some(df):
    if 'name' in df.keys():
        all_names = set(df.name.unique()) - set('Water')
    else:
        all_names = set(df['Solubility of:'].unique()) - set('Water')
    names = random.sample(list(all_names), 5)

    for name in names:
        predict_one(df, name)


class CombinedModel:
    def __init__(self, model_wia, model_aiw):
        self.model_wia = model_wia
        self.model_aiw = model_aiw

    @classmethod
    def build(cls, df, optimize=False):
        features = [feat for feat in SELECTED_FEATURES if feat not in (
            'Solubility of:', 'In:'
        )]
        df = df[features + target].dropna()
        return build_model(df, optimize)
    
    def predict(self, df):
        df_wia = df[df['Solubility of:'] == 'Water']
        df_aiw = df[df['Solubility of:'] != 'Water']

        features = [feat for feat in SELECTED_FEATURES if feat not in (
            'Solubility of:', 'In:'
        )]

        y_pred_wia = self.model_wia.predict(df_wia[features])
        y_pred_aiw = self.model_aiw.predict(df_aiw[features])
        return np.concatenate([y_pred_aiw, y_pred_wia])

def main():
    global SELECTED_FEATURES
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--predict", action="store_true", help="Show plots")
    parser.add_argument("--model", action="store_true", help="mutual solubility")
    parser.add_argument("--optimize", action="store_true", help="Run HPO")
    args = parser.parse_args()

    df = load_data()
    # For all rows with 'Solubility of: = water', replace 'x' with 1 - 'x'
    df.loc[df['Solubility of:'] == 'Water', 'x'] = 1 - df['x']

    SELECTED_FEATURES = selected_features_dual


    if args.predict:
        predict_some(df)
    else:
        build_combined_model(df, args.optimize)



if __name__ == "__main__":
    main()
