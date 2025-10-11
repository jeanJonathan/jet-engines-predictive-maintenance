import pandas as pd
import matplotlib.pyplot as plt
import os

columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_PATH = os.path.join(DATA_PATH, 'exploration')
os.makedirs(OUTPUT_PATH, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_PATH, 'train_FD001.txt'), sep=r'\s+', header=None, names=columns)
test = pd.read_csv(os.path.join(DATA_PATH, 'test_FD001.txt'), sep=r'\s+', header=None, names=columns)
rul_test = pd.read_csv(os.path.join(DATA_PATH, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])

# Create RUL labels for training data
train['RUL'] = train.groupby('unit_id')['time_cycles'].transform(lambda x: x.max() - x)

# Basic stats
print(f"\nTraining: {train.shape[0]} rows, {train['unit_id'].nunique()} engines")
print(f"Test: {test.shape[0]} rows, {test['unit_id'].nunique()} engines")
print(f"Sensors: {len([c for c in columns if 'sensor' in c])}")

# Engine lifetimes
lifetimes = train.groupby('unit_id')['time_cycles'].max()
print(f"\nEngine lifetime: min={lifetimes.min()}, max={lifetimes.max()}, avg={lifetimes.mean():.1f} cycles")

# Engine lifetimes
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(lifetimes, bins=30, edgecolor='black')
plt.xlabel('Lifetime (cycles)')
plt.ylabel('Count')
plt.title('Engine Lifetimes')

plt.subplot(1, 2, 2)
plt.boxplot(lifetimes)
plt.ylabel('Cycles')
plt.title('Lifetime Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'lifetimes.png'), dpi=100)

# Sensor degradation for engine #1
sensor_cols = [c for c in columns if 'sensor' in c]
engine1 = train[train['unit_id'] == 1]

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()
for i in range(6):
    axes[i].plot(engine1['time_cycles'], engine1[sensor_cols[i]])
    axes[i].set_title(sensor_cols[i])
    axes[i].set_xlabel('Cycles')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'sensors.png'), dpi=100)

# RUL countdown
plt.figure(figsize=(10, 5))
for i in range(1, 6):
    eng = train[train['unit_id'] == i]
    plt.plot(eng['time_cycles'], eng['RUL'], label=f'Engine {i}')
plt.xlabel('Time (cycles)')
plt.ylabel('RUL (cycles)')
plt.title('RUL Countdown')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PATH, 'rul.png'), dpi=100)

