import time
import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from iupac_scraper import parse_all
from sklearn.metrics import make_scorer
from smiles_fingerprints import compute_rdkit_features
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from config import config
from iupac_scraper import parse_all, prepare_data_for_learning
import plotly
import plotly.graph_objs as go


import matplotlib.pyplot as plt
from smiles_fingerprints import create_morgan_generator

selected_features = [
    'T',
    'aiw',  # field i added to maybe help with bimodality.
    'smiles',
]
target = ['x']

SELECTED_FEATURES = selected_features

def normalize_temperature(col):
    min = 270.15
    max = 548.15
    return (col - min) / (max - min)

def denormalize_temperature(col):
    min = 270.15
    max = 548.15
    return col * (max - min) + min

def normalize_features(df):
    """Normalize features using min-max scaling"""
    if df.empty:
        return df
    
    exclude_cols = ['aiw']
    df = df.copy()
    for col in df.columns:
        if col == 'T':
            df[col] = normalize_temperature(df[col])
            exclude_cols.append(col)
            continue
        if col == 'x':
            exclude_cols.append(col)
            continue

    scaler = MinMaxScaler()
    cols_to_scale = [col for col in df.columns if col not in exclude_cols and not col.startswith('FP_')]
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df

def select_features(df, top_N=10):
    print("Selecting features...")
    # Keep only amines in water experiments
    # df = df[df['Solubility of:'] == 'Water']

    df = df[SELECTED_FEATURES + target].dropna()
    print("Data size:", df.shape)
    print("Columns", df.columns)


    get_fingerprint = create_morgan_generator(2, 2048)
    if 'smiles' in SELECTED_FEATURES:
        fps = df['smiles'].apply(get_fingerprint)
        fps_df = pd.DataFrame(fps.apply(pd.Series).fillna(0))  # Convert sparse to fixed matrix
        fps_df.columns = [f"FP_{i}" for i in range(len(fps_df.columns))]

        df = pd.concat([df, fps_df], axis=1)

    if 'smiles' in SELECTED_FEATURES:
        smiles_features = df['smiles'].apply(compute_rdkit_features)
        smiles_features_df = pd.DataFrame(smiles_features.tolist(), index=df.index)

        df = pd.concat([df, smiles_features_df], axis=1)

    df = df.drop(columns=['smiles'])

    top_features = pd.read_csv('top_features.csv')
    top_features = top_features['feature'].tolist()[:top_N]
    df = df[df.columns.intersection(top_features + target)]

    print("Data size:", df.shape)
    print("Columns", df.columns)

    return df

def train_model_simple(X_train, y_train):    
    """Train and evaluate an XGBoost model"""

    # From testing 4/15
    xgb_optimized_hyperparameters = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 300,
        'random_state': 42,
        'subsample': 0.9,
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


def train_model_optimized(X_train, y_train):
    print("Optimizing hyperparameters...")

    xgb_param_grid = {
        "random_state": [42],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [300, 400],
        'max_depth': [4, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.7, 0.8],
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

    # grid_search = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions=param_grid,
    #     n_iter=25,  # number of random combinations to try
    #     scoring='neg_root_mean_squared_error',
    #     cv=5,
    #     n_jobs=-1,
    #     random_state=42,
    # )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
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


def plot_all_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    y_pred = maybe_unlog(y_pred)
    y_test = maybe_unlog(y_test)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(denormalize_temperature(X_test['T']), y_test, label='True', alpha=0.6, edgecolor='k', s=40)
    plt.scatter(denormalize_temperature(X_test['T']), y_pred, label='Predicted', alpha=0.6, edgecolor='k', s=40)
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


def calc_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {}

    # Metrics in log10 space
    print("Log-space metrics:")
    print("MSE: ", mean_squared_error(y_test, y_pred))
    metrics['log_mse'] = mean_squared_error(y_test, y_pred)
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics['log_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    metrics['log_mae'] = mean_absolute_error(y_test, y_pred)
    print("R2:  ", r2_score(y_test, y_pred))
    metrics['log_r2'] = r2_score(y_test, y_pred)

    # Inverse-transform predictions and ground truth
    y_test_inv = maybe_unlog(y_test)
    y_pred_inv = maybe_unlog(y_pred)

    # Metrics in original solubility space
    print("\nOriginal-space metrics:")
    print("MSE: ", mean_squared_error(y_test_inv, y_pred_inv))
    metrics['orig_mse'] = mean_squared_error(y_test_inv, y_pred_inv)
    print("RMSE:", np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    metrics['orig_rmse'] = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print("MAE: ", mean_absolute_error(y_test_inv, y_pred_inv))
    metrics['orig_mae'] = mean_absolute_error(y_test_inv, y_pred_inv)
    print("R2:  ", r2_score(y_test_inv, y_pred_inv))
    metrics['orig_r2'] = r2_score(y_test_inv, y_pred_inv)

    return metrics


def build_model(data, random_state=42):
    X = data.drop(columns=['x'])
    y = data['x']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    if config['optimize']:
        model = train_model_optimized(X_train, y_train)
    else:
        model = train_model_simple(X_train, y_train)

    feature_names = list(X.columns)
    if config['model'] == 'xgboost':
        model.get_booster().feature_names = feature_names
    elif config['model'] == 'catboost':
        model.feature_names = feature_names
    
    rmse = calc_metrics(model, X_test, y_test)
    if config['graphs']:
        shap_analysis(model, X_test)
        plot_all_predictions(model, X_test, y_test)
        plot_parity(model, X_test, y_test)
        plot_feature_importance(model)

    return model, rmse

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
    model, rmse = build_model(df_train)

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

def maybe_unlog(x):
    if config['logscale'] != 'yes':
        return x
    out = np.exp(x) - 1e-6
    return out

def maybe_log(x):
    if config['logscale'] != 'yes':
        print('not log scaling')
        return x
    
    print('log scaling')
    # Take the natural log of x + epsilon to avoid ln(0)
    return np.log(x + 1e-6)

def plot_prediction(model, name, df):
    y_pred = model.predict(df.drop(columns=['x']))
    y_actual = df['x']

    # compare the prediction and actual values w/r2
    r2 = r2_score(y_actual, y_pred)

    y_pred = maybe_unlog(y_pred)
    y_actual = maybe_unlog(y_actual)

    # Plot the prediction and actual against the temperature
    # using plotly
    fig = plotly.graph_objs.Figure()
    trace_pred = plotly.graph_objs.Scatter(
        x=y_pred,
        y=denormalize_temperature(df['T']),
        mode='markers',
        name='Predicted',
        marker=dict(
            color='blue'
        )
    )
    trace_actual = plotly.graph_objs.Scatter(
        x=y_actual,
        y=denormalize_temperature(df['T']),
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


def get_importances(model, X):
    booster = model.get_booster()
    
    # 2) XGBoost split‑count importances
    imp_dict = booster.get_score(importance_type='weight')
    fi_df = pd.DataFrame(list(imp_dict.items()), columns=['feature','importance'])
    fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # 3) SHAP importances
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_abs_shap
    })
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    return fi_df, shap_df

def load_model(path='model_iupac.json'):
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model

def load_data():
    df = parse_all()
    df = prepare_data_for_learning(df)

    df['x'] = maybe_log(df['x'])
    df = normalize_features(df)

    dfs = select_features(df, top_N = 200)
    return dfs

def find_N():
    df = parse_all()
    df = prepare_data_for_learning(df)
    df['x'] = maybe_log(df['x'])
    df = normalize_features(df)

    results = []
    for i in range(10, 1000, 10):
        dfs = select_features(df, top_N=i)
        start = time.time()
        n_metrics = [build_model(dfs.copy(), random.randint(0, 100))[1] for _ in range(5)]
        log_rmse = np.mean([m['log_rmse'] for m in n_metrics])
        end = time.time()
        elapsed = (end - start) / 5

        results.append({
            'N_bits': i,
            'rmse': log_rmse,
            'train_time_s': elapsed
        })

    perf = pd.DataFrame(results)

    # 4) Plot RMSE vs N and Training Time vs N
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(perf['N_bits'], perf['rmse'], marker='o', label='Test RMSE')
    ax1.set_xlabel('Number of fingerprint bits (N)')
    ax1.set_ylabel('Log‑space RMSE')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(perf['N_bits'], perf['train_time_s'], marker='s', color='gray', label='Training Time (s)')
    ax2.set_ylabel('Training time (s)')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('RMSE and Training Time vs Number of FP bits')
    plt.tight_layout()
    plt.show()    


def main():
    global SELECTED_FEATURES
    print(config)
    dfs = load_data()
    model, metrics = build_model(dfs)
    model.save_model('model_iupac.json')
    return model, dfs



if __name__ == "__main__":
    main()

