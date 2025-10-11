import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load preprocessed data
X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

model = keras.Sequential([
    layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=128,
    verbose=1
)

model.save(os.path.join(OUTPUT_PATH, 'lstm_model.keras'))

y_pred_train = model.predict(X_train).flatten()
y_pred_test = model.predict(X_test).flatten()
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTrain RMSE: {rmse_train:.2f} cycles")
print(f"Test RMSE: {rmse_test:.2f} cycles")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--')
plt.xlabel('Actual RUL (cycles)')
plt.ylabel('Predicted RUL (cycles)')
plt.title(f'Test Predictions (RMSE={rmse_test:.1f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_results.png'), dpi=100)
