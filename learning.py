import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import shap
import matplotlib.pyplot as plt

from config import config
from amine_solubility import load_data, load_mutual_solubility_data
import plotly
import plotly.graph_objs as go


import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from smiles_fingerprints import create_morgan_generator

selected_features_orig = [
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
]

selected_features_dual = [
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

    # 'Solute Fingerprint',
    # 'Solvent Fingerprint',
]


selected_features_one_compound = [
    'T (K)',
    'molecular_weight_gpm',
    'xlogp3_aa',
    'hydrogen_bond_donor_count',
    'hydrogen_bond_acceptor_count',
    'rotatable_bond_count',
    # 'exact_mass_da',
    # 'monoisotopic_mass_da',
    'topological_polar_surface_area_angstroms',
    # 'heavy_atom_count',
    'complexity',
    'undefined_atom_stereocenter_count',
    'aiw',  # field i added to maybe help with bimodality.
#    'smiles',  # file i added to add more structural features.
               #  Currently only works for ms model.
]

target = ['x']

SELECTED_FEATURES = selected_features_one_compound


def pearson_correlation_coefficient(df):
    # omit the 'FP_' columns
    df = df.drop(columns=[col for col in df.columns if 'FP_' in col])
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Feature Correlation Matrix")
    plt.show()


def vif(df):
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


    get_fingerprint = create_morgan_generator(2, 100)
    if 'smiles' in SELECTED_FEATURES:
        fps = df['smiles'].apply(get_fingerprint)
        fps_df = pd.DataFrame(fps.apply(pd.Series).fillna(0))  # Convert sparse to fixed matrix
        fps_df.columns = [f"FP_{i}" for i in range(len(fps_df.columns))]
        df = pd.concat([df, fps_df], axis=1)
        df = df.drop(columns=['smiles'])

    return df

def train_model_simple(X_train, y_train):    
    """Train and evaluate an XGBoost model"""

    # From testing 2/17
    xgb_optimized_hyperparameters = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 400,
        'random_state': 42,
        'subsample': 0.8,
    }

    # 2/17
    catboost_hyperparameters = {
        'bagging_temperature': 0.7,
        'depth': 5,
        'eta': 0.1,
        'iterations': 400,
        'random_state': 42,
        'rsm': 0.9,
    }
    if config['model'] == 'catboost':
        print('using catboost')
        model = CatBoostRegressor(verbose=False, **catboost_hyperparameters)
    elif config['model'] == 'xgboost':
        print('using xgboost')
        model = xgb.XGBRegressor(
            **xgb_optimized_hyperparameters
        )
    
    model.fit(X_train, y_train)
    return model

def pseudohuber_loss(y_true, y_pred, delta=1.0):
    """Computes Pseudo-Huber loss"""
    residual = y_true - y_pred
    return np.mean(delta**2 * (np.sqrt(1 + (residual / delta) ** 2) - 1))

# Convert the function into a scorer for GridSearchCV
pseudohuber_scorer = make_scorer(pseudohuber_loss, greater_is_better=False)  # Lower loss is better



def train_model_optimized(X_train, y_train):
    print("Optimizing hyperparameters...")

    xgb_param_grid = {
        "random_state": [42],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        # 'objective': ['reg:pseudohubererror'],
    }

    catboost_param_grid = {
        "random_state": [42],
        'eta': [0.001, 0.01, 0.1],  # equivalent to learning rate
        'iterations': [200, 300, 400],  # equivalent to n_estimators
        'depth': [3, 4, 5],  # equivalent to max depth
        'bagging_temperature': [0.7, 0.8, 0.9], # equivalent to subsample
        'rsm': [0.7, 0.8, 0.9],  # equivalent to colsample_bytree
    }

    if config['model'] == 'catboost':
        print('Using catboost...')
        param_grid = catboost_param_grid
        model = CatBoostRegressor(verbose=0, **param_grid)
    elif config['model'] == 'xgboost':
        print('Using xgboost...')
        param_grid = xgb_param_grid
        model = xgb.XGBRegressor(**param_grid)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        # scoring=pseudohuber_scorer,  # Use the custom loss function
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
    if config['model'] == 'catboost':
        importances = model.get_feature_importance()  # Loss-based importance
        feature_names = model.feature_names
        indices = np.argsort(importances)[::-1]  # Sort in descending order

        plt.figure(figsize=(10, 6))
        plt.barh(range(20), importances[indices][:20], align="center")
        plt.yticks(range(20), np.array(feature_names)[indices[:20]], rotation=0)
        plt.gca().invert_yaxis()  # Flip the y-axis
        plt.ylabel("Feature")
        plt.xlabel("Importance")
        plt.title("CatBoost Feature Importance")
        plt.tight_layout()
        plt.show()
    else:
        xgb.plot_importance(model,
                            importance_type='weight',
                            show_values=False,
                            max_num_features=20,)
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

def build_model(data):
    X = data.drop(columns=['x'])
    y = data['x']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if config['optimize']:
        model = train_model_optimized(X_train, y_train)
    else:
        model = train_model_simple(X_train, y_train)

    feature_names = list(X.columns)
    if config['model'] == 'xgboost':
        model.get_booster().feature_names = feature_names
    elif config['model'] == 'catboost':
        model.feature_names = feature_names
    
    print_metrics(model, X_test, y_test)
    if config['graphs']:
        shap_analysis(model, X_test)
        pearson_correlation_coefficient(data)
        plot_predictions(model, X_test, y_test)
        plot_parity(model, X_test, y_test)
        plot_feature_importance(model)
    return model

def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])

def predict_some(df, names=None):
    if 'name' in df.keys():
        all_names = set(df.name.unique()) - set('Water')
    else:
        all_names = set(df['Solubility of:'].unique()) - set('Water')

    if names is None:
        names = random.sample(list(all_names), 5)
        names += ['Diisopropylamine (C6H15N)', 'Dipropylamine (C6H15N)']

    # partition the df into two parts depending on a condition
    if 'name' in df.keys():
        cond = (df['name'].isin(names))
    else:
        cond = (df['Solubility of:'].isin(names)) | (df['In:'].isin(names))

    name_not_matches = df[~cond]
    df_train = select_features(name_not_matches)
    model = build_model(df_train)

    df_test_by_name = {}
    for name in names:
        if 'name' in df.keys():
            cond = (df['name'] == name)
        else:
            cond = (df['Solubility of:'] == name) | (df['In:'] == name)

        df_test = select_features(df[cond])
        if df_test.empty:
            print(f'No test points for {name}')
            continue
        df_test_by_name[name] = df_test

    for name, df_test in df_test_by_name.items():
        yield model, name, df_test

def unlog(x):
    out = np.exp(x) - 1e-6
    return out

def log(x):
    # Take the natural log of x + epsilon to avoid ln(0)
    return np.log(x + 1e-6)

def plot_prediction(model, name, df):
    y_pred = model.predict(df.drop(columns=['x']))
    y_actual = df['x']

    # compare the prediction and actual values w/r2
    r2 = r2_score(y_actual, y_pred)

    # Plot the prediction and actual against the temperature
    # using plotly
    fig = plotly.graph_objs.Figure()
    trace_pred = plotly.graph_objs.Scatter(
        x=y_pred,
        y=df['T (K)'],
        mode='markers',
        name='Predicted',
        marker=dict(
            color='blue'
        )
    )
    trace_actual = plotly.graph_objs.Scatter(
        x=y_actual,
        y=df['T (K)'],
        mode='markers',
        name='Actual',
        marker=dict(
            color='red'
        )
    )
    fig.layout.title = name + (' (R2: %.2f)' % r2)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[250, 510])
    fig.add_trace(trace_pred)
    fig.add_trace(trace_actual)
    fig.show()

def test_per_amine(df):
    names = set()
    for key in ['name', 'Solubility of:', 'In:']:
        if key in df.keys():
            names |= set(df[key].unique())

    dfs = []
    for name in names:
        for model, name, test_df in predict_some(df, [name]):
            y_pred = model.predict(test_df.drop(columns=['x']))
            y_actual = test_df['x']
            temp = test_df['T (K)']
            r2 = r2_score(y_actual, y_pred)
            dfs.append(pd.DataFrame({
                'name': name,
                'r2': r2,
                'T (K)': temp.tolist(),
                'y_pred': y_pred.tolist(), 
                'y_actual': y_actual.tolist()
            }))

    out = pd.concat(dfs)
    out.to_csv('data/per_amine.csv', index=False)
    return out

def main():
    global SELECTED_FEATURES
    print(config)

    if config['features'] == 'ms':
        df = load_mutual_solubility_data()
        SELECTED_FEATURES = selected_features_one_compound
    elif config['features'] == 'ms-smiles':
        df = load_mutual_solubility_data()
        SELECTED_FEATURES = selected_features_one_compound + ['smiles']
    else:
        df = load_data()
        # For all rows with 'Solubility of: = water', replace 'x' with 1 - 'x'
        df.loc[df['Solubility of:'] == 'Water', 'x'] = 1 - df['x']
        SELECTED_FEATURES = selected_features_dual

    if config['predict']:
        for model, name, df in predict_some(df):
            plot_prediction(model, name, df)
        return

    if config['test_per_amine']:
        test_per_amine(df)
        return

    build_model(select_features(df))



if __name__ == "__main__":
    main()
