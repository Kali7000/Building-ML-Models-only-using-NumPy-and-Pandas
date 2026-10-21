# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:52:38 2026

@author: Kali
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(df):
    stats = {
        'mean': df.mean(),
        'std':  df.std()
    }
    df_z = (df - stats['mean']) / stats['std']
    return df_z, stats


def normalize_with_stat(df, stat):
    # Uses the stats from training data — never refit on test
    df_z = (df - stat['mean']) / stat['std']
    return df_z
   

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preProcessing(file_name, ordinal_encoding_threshold, ordinal_maps=None):
    """
    ordinal_maps=None  → training mode: builds ordinal maps from this data and returns them.
    ordinal_maps=dict  → test mode: reuses the maps built from training data.

    This is critical. If maps are rebuilt from test data the encoding order can
    differ from training, feeding the model nonsense feature values.
    """
    df = pd.read_csv(file_name)
    column_list = df.columns.tolist()

    str_cols, int_cols = [], []
    for col in column_list:
        (str_cols if df[col].dtype == 'object' else int_cols).append(col)

    df = df.dropna(subset=int_cols)
    id_df = df[['Id']]
    df = df.drop(columns=['Id'])

    oneHot_col  = []
    built_maps  = {}

    for obj_col in str_cols:
        if obj_col not in df.columns:
            continue
        unique_vals = df[obj_col].unique().tolist()

        if len(unique_vals) >= ordinal_encoding_threshold:
            if ordinal_maps is not None:
                # Test mode — use the map from training
                mapping = ordinal_maps.get(obj_col, {})
            else:
                # Train mode — build the map once and save it
                mapping = {val: i + 1 for i, val in enumerate(unique_vals)}
                built_maps[obj_col] = mapping

            df[obj_col] = df[obj_col].map(mapping)
        else:
            oneHot_col.append(obj_col)

    df_processed = pd.get_dummies(df, columns=oneHot_col, drop_first=True, dtype=float)

    if ordinal_maps is None:
        return df_processed, built_maps   # training: return maps
    return df_processed, id_df                   # test: no maps to return


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(X):
    """
    Builds degree-2 polynomial feature matrix from a (n, d) NumPy array:
      [linear terms | squared terms | pairwise cross terms]
    Runs once on the full dataset — not row-by-row.
    """
    n, d = X.shape
    X2   = X ** 2
    cross_idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if cross_idx:
        cross  = np.column_stack([X[:, i] * X[:, j] for i, j in cross_idx])
        X_feat = np.hstack([X, X2, cross])
    else:
        X_feat = np.hstack([X, X2])
    return X_feat


def normalize_features(X_feat, feat_mean=None, feat_std=None):
    """
    Normalizes the full feature matrix.
    Pass feat_mean/feat_std from training when normalizing test features.

    Degree-2 features (squares, cross terms) have a much larger range than
    the original normalized inputs (~[-3,3] → squares ~[0,9], cross ~[-9,9]).
    Without this second normalization those features dominate the gradient
    and constantly hit the ±5 clip, killing their learning signal.
    """
    if feat_mean is None:
        feat_mean = X_feat.mean(axis=0)
        feat_std  = X_feat.std(axis=0)
    # avoid division by zero for constant columns
    feat_std_safe = np.where(feat_std < 1e-8, 1.0, feat_std)
    return (X_feat - feat_mean) / feat_std_safe, feat_mean, feat_std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lasso(train_df, lambda_val=0.5, learning_rate=0.001,
                epochs=1000, patience=10):
    """
    Vectorised batch gradient descent with Lasso (L1) regularisation.

    Key changes vs. the original:
      - build_feature_matrix runs once before the loop (not per row)
      - gradients are matrix ops on the full dataset (no Python inner loop)
      - early stopping requires `patience` consecutive non-improving checkpoints
        instead of stopping after just one bad epoch
      - bias is updated with the mean gradient (no L1 penalty — intentional)
    """
    y     = train_df.pop('SalePrice').values        # (n,)
    X_raw = train_df.values                         # (n, d)

    X_feat, feat_mean, feat_std = normalize_features(build_feature_matrix(X_raw))

    n_samples, n_features = X_feat.shape
    weights = np.zeros(n_features)
    bias    = 0.0

    best_mse         = None
    no_improve_count = 0

    for epoch in range(epochs):
        # --- Forward pass (whole dataset in one matmul) ---
        preds  = X_feat @ weights + bias    # (n,)
        errors = preds - y                  # (n,)

        # --- Gradient for weights (L1 penalty applied to weights only) ---
        grad_w  = (X_feat.T @ errors) / n_samples   # (F,)
        grad_w += lambda_val * np.sign(weights)      # Lasso penalty
        grad_w  = np.clip(grad_w, -5.0, 5.0)

        weights -= learning_rate * grad_w
        weights[np.abs(weights) < 1e-4] = 0.0       # snap-to-zero

        # --- Gradient for bias (no L1 penalty) ---
        grad_b = np.clip(np.mean(errors), -5.0, 5.0)
        bias  -= learning_rate * grad_b

        # --- Logging + early stopping ---
        if epoch % 10 == 0:
            mse = np.mean(errors ** 2)
            print(f"Epoch {epoch:>4d}: train MSE = {mse:.6f}")

            if best_mse is None or mse < best_mse:
                best_mse         = mse
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch}. "
                          f"Best MSE: {best_mse:.6f}")
                    break

    return weights, bias, feat_mean, feat_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ordinal_encoding_threshold = 3

# --- Train ---
train_df, ordinal_maps = preProcessing('train.csv', ordinal_encoding_threshold)

train_df, z_parameters = normalize(train_df)

salePriceMean = z_parameters['mean'].pop('SalePrice')
salePriceSTD  = z_parameters['std'].pop('SalePrice')


def deNormalize(val):
    return (val * salePriceSTD) + salePriceMean


lambda_val    = 0.002
learning_rate = 0.001
epochs        = 1000

weights, bias, feat_mean, feat_std = train_lasso(
    train_df, lambda_val, learning_rate, epochs
)

# --- Test ---
# Pass ordinal_maps so test encoding matches training exactly
test_df, id_df = preProcessing('test.csv', ordinal_encoding_threshold,
                               ordinal_maps=ordinal_maps)

# Normalise test features using training statistics
test_df = normalize_with_stat(test_df, z_parameters)

# Align columns so test has the same dummy columns as train
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0.0)

# Build and normalise the degree-2 feature matrix using training mean/std
X_test, _, _ = normalize_features(
    build_feature_matrix(test_df.values), feat_mean, feat_std
)

# Predict in one vectorised pass
raw_preds          = X_test @ weights + bias
id_df['SalePrice'] = [deNormalize(p) for p in raw_preds]