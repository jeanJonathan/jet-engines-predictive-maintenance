"""
02_preprocessing.py
====================
Feature Engineering & Sequence Preprocessing — NASA C-MAPSS FD001
Author : Jean-Jonathan KOFFI

Improvements over baseline:
  - Capped RUL (piecewise linear) — standard in literature (Heimes 2008)
  - Rolling statistics as additional features (mean & std over window)
  - Variance-based sensor selection
  - RobustScaler instead of MinMax (less sensitive to outliers)
  - Configurable sequence length with validation
  - Full train/val/test split with stratification by engine
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

# ─── Config ───────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH   = 30          # Look-back window (cycles)
RUL_CAP           = 125         # Piecewise-linear RUL cap (Heimes 2008)
VARIANCE_THRESHOLD = 0.01       # Drop sensors below this variance
ROLLING_WINDOW    = 5           # Window for rolling statistics features
VAL_ENGINES       = 20          # Number of engines held out for validation

BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH   = os.path.join(BASE_DIR, 'data')
OUTPUT_PATH = os.path.join(DATA_PATH, 'processed')
os.makedirs(OUTPUT_PATH, exist_ok=True)

COLUMNS = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
OP_COLS     = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
train    = pd.read_csv(os.path.join(DATA_PATH, 'train_FD001.txt'), sep=r'\s+', header=None, names=COLUMNS)
test     = pd.read_csv(os.path.join(DATA_PATH, 'test_FD001.txt'),  sep=r'\s+', header=None, names=COLUMNS)
rul_test = pd.read_csv(os.path.join(DATA_PATH, 'RUL_FD001.txt'),   sep=r'\s+', header=None, names=['RUL'])

# ─── RUL labels (capped — piecewise linear) ───────────────────────────────────
print(f"Computing RUL labels (cap={RUL_CAP} cycles)...")
train['RUL'] = train.groupby('unit_id')['time_cycles'].transform(lambda x: x.max() - x)
train['RUL'] = train['RUL'].clip(upper=RUL_CAP)   # ← key improvement

# ─── Sensor selection ─────────────────────────────────────────────────────────
variances     = train[SENSOR_COLS].var()
drop_sensors  = variances[variances < VARIANCE_THRESHOLD].index.tolist()
useful_sensors = [s for s in SENSOR_COLS if s not in drop_sensors]
print(f"Dropped {len(drop_sensors)} constant sensors: {drop_sensors}")
print(f"Keeping {len(useful_sensors)} sensors + {len(OP_COLS)} op settings")

# ─── Rolling statistics features ──────────────────────────────────────────────
print(f"Engineering rolling features (window={ROLLING_WINDOW})...")

def add_rolling_features(df, sensors, window):
    df = df.copy().sort_values(['unit_id', 'time_cycles'])
    for sensor in sensors:
        grp = df.groupby('unit_id')[sensor]
        df[f'{sensor}_mean{window}'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{sensor}_std{window}']  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return df

train = add_rolling_features(train, useful_sensors, ROLLING_WINDOW)
test  = add_rolling_features(test,  useful_sensors, ROLLING_WINDOW)

# ─── Feature columns ──────────────────────────────────────────────────────────
rolling_means = [f'{s}_mean{ROLLING_WINDOW}' for s in useful_sensors]
rolling_stds  = [f'{s}_std{ROLLING_WINDOW}'  for s in useful_sensors]
feature_cols  = OP_COLS + useful_sensors + rolling_means + rolling_stds

print(f"Total features: {len(feature_cols)}")

# ─── Normalization (RobustScaler) ─────────────────────────────────────────────
scaler = RobustScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols]  = scaler.transform(test[feature_cols])
joblib.dump(scaler, os.path.join(OUTPUT_PATH, 'scaler.pkl'))
print("Scaler saved.")

# ─── Train / Validation split by engine ───────────────────────────────────────
all_engines  = train['unit_id'].unique()
np.random.seed(42)
val_engines  = np.random.choice(all_engines, size=VAL_ENGINES, replace=False)
train_engines = [e for e in all_engines if e not in val_engines]

df_train = train[train['unit_id'].isin(train_engines)]
df_val   = train[train['unit_id'].isin(val_engines)]
print(f"Train engines: {len(train_engines)} | Val engines: {len(val_engines)}")

# ─── Sequence creation ────────────────────────────────────────────────────────
def create_sequences(df, seq_len, feature_cols):
    X, y = [], []
    for uid in df['unit_id'].unique():
        eng     = df[df['unit_id'] == uid].sort_values('time_cycles')
        feats   = eng[feature_cols].values
        rul_vals = eng['RUL'].values
        for i in range(len(feats) - seq_len + 1):
            X.append(feats[i:i+seq_len])
            y.append(rul_vals[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_test_sequences(df, seq_len, feature_cols):
    X = []
    for uid in sorted(df['unit_id'].unique()):
        eng   = df[df['unit_id'] == uid].sort_values('time_cycles')
        feats = eng[feature_cols].values
        if len(feats) < seq_len:
            pad   = np.zeros((seq_len - len(feats), len(feature_cols)), dtype=np.float32)
            feats = np.vstack([pad, feats])
        X.append(feats[-seq_len:])
    return np.array(X, dtype=np.float32)

print("Creating sequences...")
X_train, y_train = create_sequences(df_train, SEQUENCE_LENGTH, feature_cols)
X_val,   y_val   = create_sequences(df_val,   SEQUENCE_LENGTH, feature_cols)
X_test           = create_test_sequences(test, SEQUENCE_LENGTH, feature_cols)

# Clip test RUL labels too
y_test = np.minimum(rul_test['RUL'].values.astype(np.float32), RUL_CAP)

print(f"\n{'='*50}")
print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")
print(f"{'='*50}\n")

# ─── Save ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_PATH, 'X_val.npy'),   X_val)
np.save(os.path.join(OUTPUT_PATH, 'y_val.npy'),   y_val)
np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'),  X_test)
np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'),  y_test)

# Save metadata
import json
meta = {
    'sequence_length': SEQUENCE_LENGTH,
    'rul_cap': RUL_CAP,
    'feature_cols': feature_cols,
    'n_features': len(feature_cols),
    'useful_sensors': useful_sensors,
    'dropped_sensors': drop_sensors,
}
with open(os.path.join(OUTPUT_PATH, 'metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("All preprocessed data saved to:", OUTPUT_PATH)