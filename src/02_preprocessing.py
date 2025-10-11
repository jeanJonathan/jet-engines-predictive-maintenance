import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_PATH = os.path.join(DATA_PATH, 'processed')
os.makedirs(OUTPUT_PATH, exist_ok=True)

columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

train = pd.read_csv(os.path.join(DATA_PATH, 'train_FD001.txt'), sep=r'\s+', header=None, names=columns)
test = pd.read_csv(os.path.join(DATA_PATH, 'test_FD001.txt'), sep=r'\s+', header=None, names=columns)
rul_test = pd.read_csv(os.path.join(DATA_PATH, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])

train['RUL'] = train.groupby('unit_id')['time_cycles'].transform(lambda x: x.max() - x)

# Drop constant sensors (variance < 0.01)
sensor_cols = [c for c in columns if 'sensor' in c]
variances = train[sensor_cols].var()
drop_sensors = variances[variances < 0.01].index.tolist()
print(f"Dropping {len(drop_sensors)} constant sensors: {drop_sensors}")

# Keep useful features
feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [s for s in sensor_cols if s not in drop_sensors]
print(f"Using {len(feature_cols)} features")

# Normalize features (0-1)
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# Save scaler
joblib.dump(scaler, os.path.join(OUTPUT_PATH, 'scaler.pkl'))

# Create sequences for LSTM (window size = 30 cycles)
SEQUENCE_LENGTH = 30

def create_sequences(df, seq_len):
    sequences = []
    targets = []

    for unit_id in df['unit_id'].unique():
        engine_data = df[df['unit_id'] == unit_id]
        features = engine_data[feature_cols].values
        rul_values = engine_data['RUL'].values

        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i+seq_len])
            targets.append(rul_values[i+seq_len-1])

    return np.array(sequences), np.array(targets)

X_train, y_train = create_sequences(train, SEQUENCE_LENGTH)
print(f"Train: {X_train.shape}, {y_train.shape}")

# For test data, use last sequence of each engine
X_test = []
y_test = rul_test['RUL'].values

for unit_id in test['unit_id'].unique():
    engine_data = test[test['unit_id'] == unit_id]
    features = engine_data[feature_cols].values

    # Pad if less than sequence length
    if len(features) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(features), len(feature_cols)))
        features = np.vstack([pad, features])

    X_test.append(features[-SEQUENCE_LENGTH:])

X_test = np.array(X_test)
print(f"Test: {X_test.shape}, {y_test.shape}")

# Save preprocessed data
np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)