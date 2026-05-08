"""
03_train_model.py
==================
Multi-Model Benchmark — NASA C-MAPSS FD001
Author : Jean-Jonathan KOFFI

Models trained and compared:
  1. LSTM (Bidirectional) — deep learning baseline
  2. XGBoost Regressor    — gradient boosting on flattened sequences
  3. Random Forest        — ensemble baseline

Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE  (Mean Absolute Error)
  - NASA Scoring Function (asymmetric — penalises late predictions more)
  - R²   (Coefficient of determination)

Outputs:
  - data/models/lstm_model.keras
  - data/models/xgboost_model.pkl
  - data/models/rf_model.pkl
  - data/models/benchmark_results.json
  - data/models/benchmark_plot.png
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH  = os.path.join(BASE_DIR, 'data', 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_PATH, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_PATH, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

print(f"  X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")

N_TIMESTEPS = X_train.shape[1]
N_FEATURES  = X_train.shape[2]

# ─── NASA Scoring Function ────────────────────────────────────────────────────
def nasa_score(y_true, y_pred):
    """
    Asymmetric scoring function from the PHM08 challenge.
    Penalises late predictions (predicted RUL > actual) more than early ones.
    Lower is better. Score = sum(exp(d/13) - 1  if d < 0
                                  exp(d/10) - 1  if d >= 0)
    where d = y_pred - y_true
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))

def compute_metrics(y_true, y_pred, label=""):
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    score = nasa_score(y_true, y_pred)
    if label:
        print(f"  [{label}] RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.3f} | NASA Score={score:.0f}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nasa_score': score}

# ─── Flatten helper for non-sequential models ─────────────────────────────────
def flatten(X):
    return X.reshape(X.shape[0], -1)

# ─── 1. LSTM (Bidirectional) ──────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL 1 : Bidirectional LSTM")
print("="*55)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_lstm(n_timesteps, n_features):
    inp = keras.Input(shape=(n_timesteps, n_features))
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

lstm_model = build_lstm(N_TIMESTEPS, N_FEATURES)
lstm_model.summary()

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5, monitor='val_loss')
]

t0 = time.time()
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)
lstm_time = time.time() - t0

lstm_model.save(os.path.join(MODEL_PATH, 'lstm_model.keras'))
lstm_pred_test  = lstm_model.predict(X_test).flatten()
lstm_pred_train = lstm_model.predict(X_train).flatten()

print("\nLSTM Results:")
lstm_train_metrics = compute_metrics(y_train, lstm_pred_train, "Train")
lstm_test_metrics  = compute_metrics(y_test,  lstm_pred_test,  "Test ")

# ─── 2. XGBoost ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL 2 : XGBoost Regressor")
print("="*55)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("  XGBoost not installed — skipping (pip install xgboost)")
    HAS_XGBOOST = False

if HAS_XGBOOST:
    X_train_flat = flatten(X_train)
    X_val_flat   = flatten(X_val)
    X_test_flat  = flatten(X_test)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric='rmse',
        verbosity=1,
    )

    t0 = time.time()
    xgb_model.fit(X_train_flat, y_train,
                  eval_set=[(X_val_flat, y_val)],
                  verbose=50)
    xgb_time = time.time() - t0

    joblib.dump(xgb_model, os.path.join(MODEL_PATH, 'xgboost_model.pkl'))
    xgb_pred_test  = xgb_model.predict(X_test_flat)
    xgb_pred_train = xgb_model.predict(X_train_flat)

    print("\nXGBoost Results:")
    xgb_train_metrics = compute_metrics(y_train, xgb_pred_train, "Train")
    xgb_test_metrics  = compute_metrics(y_test,  xgb_pred_test,  "Test ")
else:
    xgb_test_metrics = None; xgb_pred_test = None; xgb_time = 0

# ─── 3. Random Forest ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL 3 : Random Forest Regressor")
print("="*55)

X_train_flat = flatten(X_train)
X_test_flat  = flatten(X_test)

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

t0 = time.time()
rf_model.fit(X_train_flat, y_train)
rf_time = time.time() - t0

joblib.dump(rf_model, os.path.join(MODEL_PATH, 'rf_model.pkl'))
rf_pred_test  = rf_model.predict(X_test_flat)
rf_pred_train = rf_model.predict(X_train_flat)

print("\nRandom Forest Results:")
rf_train_metrics = compute_metrics(y_train, rf_pred_train, "Train")
rf_test_metrics  = compute_metrics(y_test,  rf_pred_test,  "Test ")

# ─── 4. Benchmark Summary Plot ────────────────────────────────────────────────
print("\n" + "="*55)
print("  BENCHMARK COMPARISON")
print("="*55)

models_data = {'LSTM (BiDir)': (lstm_test_metrics, lstm_pred_test, lstm_time)}
if HAS_XGBOOST and xgb_test_metrics:
    models_data['XGBoost'] = (xgb_test_metrics, xgb_pred_test, xgb_time)
models_data['Random Forest'] = (rf_test_metrics, rf_pred_test, rf_time)

model_names = list(models_data.keys())
rmses  = [models_data[m][0]['rmse']  for m in model_names]
maes   = [models_data[m][0]['mae']   for m in model_names]
r2s    = [models_data[m][0]['r2']    for m in model_names]
scores = [models_data[m][0]['nasa_score'] for m in model_names]
preds  = [models_data[m][1] for m in model_names]
times  = [models_data[m][2] for m in model_names]

colors = ['#2196F3', '#FF9800', '#4CAF50'][:len(model_names)]

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# RMSE
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(model_names, rmses, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_title('RMSE (lower=better)', fontweight='bold')
ax1.set_ylabel('cycles')
for i, v in enumerate(rmses): ax1.text(i, v+0.3, f'{v:.1f}', ha='center', fontsize=9)

# MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(model_names, maes, color=colors, edgecolor='white', linewidth=1.5)
ax2.set_title('MAE (lower=better)', fontweight='bold')
ax2.set_ylabel('cycles')
for i, v in enumerate(maes): ax2.text(i, v+0.3, f'{v:.1f}', ha='center', fontsize=9)

# R²
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(model_names, r2s, color=colors, edgecolor='white', linewidth=1.5)
ax3.set_title('R² (higher=better)', fontweight='bold')
ax3.set_ylabel('R²')
ax3.set_ylim([0, 1])
for i, v in enumerate(r2s): ax3.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)

# NASA Score
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(model_names, scores, color=colors, edgecolor='white', linewidth=1.5)
ax4.set_title('NASA PHM Score (lower=better)', fontweight='bold')
ax4.set_ylabel('Score')
for i, v in enumerate(scores): ax4.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

# Training time
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(model_names, times, color=colors, edgecolor='white', linewidth=1.5)
ax5.set_title('Training Time (s)', fontweight='bold')
ax5.set_ylabel('seconds')
for i, v in enumerate(times): ax5.text(i, v+0.5, f'{v:.0f}s', ha='center', fontsize=9)

# Scatter plots: actual vs predicted
for col, (name, pred) in enumerate(zip(model_names, preds)):
    ax = fig.add_subplot(gs[2, col]) if col < 3 else None
    if ax is None: continue
    max_val = max(y_test.max(), pred.max())
    ax.scatter(y_test, pred, alpha=0.4, s=10, color=colors[col])
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'{name} — Actual vs Predicted')
    ax.legend(fontsize=7)

# LSTM training history (replace time bar with history)
ax5b = fig.add_subplot(gs[1, 2])
ax5b.plot(history.history['loss'],     label='Train Loss', color='steelblue')
ax5b.plot(history.history['val_loss'], label='Val Loss',   color='tomato', linestyle='--')
ax5b.set_title('LSTM Training History', fontweight='bold')
ax5b.set_xlabel('Epoch')
ax5b.set_ylabel('MSE Loss')
ax5b.legend(fontsize=8)
ax5b.grid(True, alpha=0.3)

fig.suptitle('Model Benchmark — NASA C-MAPSS FD001 (RUL Prediction)', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(MODEL_PATH, 'benchmark_plot.png'), dpi=130, bbox_inches='tight')
plt.close()
print("benchmark_plot.png saved")

# ─── Save benchmark JSON ──────────────────────────────────────────────────────
benchmark = {}
for name, (metrics, pred, t) in models_data.items():
    benchmark[name] = {**metrics, 'training_time_sec': round(t, 1)}

with open(os.path.join(MODEL_PATH, 'benchmark_results.json'), 'w') as f:
    json.dump(benchmark, f, indent=2)

# Save all predictions
np.save(os.path.join(MODEL_PATH, 'lstm_pred_test.npy'),  lstm_pred_test)
np.save(os.path.join(MODEL_PATH, 'rf_pred_test.npy'),    rf_pred_test)
if HAS_XGBOOST and xgb_pred_test is not None:
    np.save(os.path.join(MODEL_PATH, 'xgb_pred_test.npy'), xgb_pred_test)

print("\n" + "="*55)
print("  FINAL BENCHMARK SUMMARY (TEST SET)")
print("="*55)
print(f"  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'NASA':>10}")
print(f"  {'-'*56}")
for name, (m, _, t) in models_data.items():
    print(f"  {name:<20} {m['rmse']:>8.2f} {m['mae']:>8.2f} {m['r2']:>8.3f} {m['nasa_score']:>10.0f}")
print("="*55)