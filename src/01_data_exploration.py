"""
01_data_exploration.py
======================
Exploratory Data Analysis — NASA C-MAPSS FD001 Dataset
Author : Jean-Jonathan KOFFI
Context: Predictive Maintenance for Turbofan Engines

Generates:
  - Engine lifetime distribution
  - Sensor degradation curves
  - Correlation heatmap (sensors vs RUL)
  - PCA of sensor space
  - Sensor variance analysis (keep vs drop)
  - RUL distribution per degradation stage
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH   = os.path.join(BASE_DIR, 'data')
OUTPUT_PATH = os.path.join(DATA_PATH, 'exploration')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── Column names ─────────────────────────────────────────────────────────────
COLUMNS = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
OP_COLS     = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading C-MAPSS FD001 dataset...")
train = pd.read_csv(os.path.join(DATA_PATH, 'train_FD001.txt'), sep=r'\s+', header=None, names=COLUMNS)
test  = pd.read_csv(os.path.join(DATA_PATH, 'test_FD001.txt'),  sep=r'\s+', header=None, names=COLUMNS)
rul_test = pd.read_csv(os.path.join(DATA_PATH, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])

# ─── RUL labels ───────────────────────────────────────────────────────────────
train['RUL'] = train.groupby('unit_id')['time_cycles'].transform(lambda x: x.max() - x)

# ─── Summary ──────────────────────────────────────────────────────────────────
lifetimes = train.groupby('unit_id')['time_cycles'].max()
print(f"\n{'='*55}")
print(f"  Training set : {train.shape[0]:>6,} rows | {train['unit_id'].nunique():>3} engines")
print(f"  Test set     : {test.shape[0]:>6,} rows  | {test['unit_id'].nunique():>3} engines")
print(f"  Sensors      : {len(SENSOR_COLS)}")
print(f"  Lifetime min : {lifetimes.min()} cycles")
print(f"  Lifetime max : {lifetimes.max()} cycles")
print(f"  Lifetime avg : {lifetimes.mean():.1f} cycles")
print(f"{'='*55}\n")

# ─── 1. Engine Lifetime Distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Engine Lifetime Distribution — FD001', fontsize=13, fontweight='bold')

axes[0].hist(lifetimes, bins=25, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(lifetimes.mean(), color='tomato', linewidth=2, linestyle='--', label=f'Mean: {lifetimes.mean():.0f} cy')
axes[0].set_xlabel('Lifetime (cycles)')
axes[0].set_ylabel('Count')
axes[0].set_title('Histogram')
axes[0].legend()

bp = axes[1].boxplot(lifetimes, patch_artist=True, vert=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['medians'][0].set_color('tomato')
axes[1].set_ylabel('Cycles')
axes[1].set_title('Box Plot')
axes[1].set_xticklabels(['All engines'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_lifetimes.png'), dpi=120)
plt.close()
print("[01_lifetimes.png")

# ─── 2. Sensor Variance Analysis ──────────────────────────────────────────────
variances = train[SENSOR_COLS].var().sort_values(ascending=False)
VARIANCE_THRESHOLD = 0.01
useful = variances[variances >= VARIANCE_THRESHOLD]
noisy  = variances[variances < VARIANCE_THRESHOLD]

fig, ax = plt.subplots(figsize=(13, 4))
colors = ['steelblue' if v >= VARIANCE_THRESHOLD else 'lightcoral' for v in variances.values]
ax.bar(variances.index, variances.values, color=colors)
ax.axhline(VARIANCE_THRESHOLD, color='tomato', linestyle='--', linewidth=1.5, label=f'Threshold={VARIANCE_THRESHOLD}')
ax.set_yscale('log')
ax.set_xlabel('Sensor')
ax.set_ylabel('Variance (log scale)')
ax.set_title('Sensor Variance — Useful vs Constant')
ax.legend()
handles = [plt.Rectangle((0,0),1,1, color='steelblue'), plt.Rectangle((0,0),1,1, color='lightcoral')]
ax.legend(handles + [plt.Line2D([0],[0], color='tomato', linestyle='--')],
          [f'Useful ({len(useful)})', f'Dropped ({len(noisy)})', f'Threshold={VARIANCE_THRESHOLD}'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_sensor_variance.png'), dpi=120)
plt.close()
print(f"02_sensor_variance.png  |  Useful: {list(useful.index)}  |  Dropped: {list(noisy.index)}")

# ─── 3. Sensor Degradation — key sensors on engine #1 ────────────────────────
useful_sensors = useful.index.tolist()
engine1 = train[train['unit_id'] == 1].sort_values('time_cycles')

n = min(len(useful_sensors), 12)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
axes = axes.flatten()

for i, sensor in enumerate(useful_sensors[:n]):
    ax = axes[i]
    ax.plot(engine1['time_cycles'], engine1[sensor], color='steelblue', linewidth=1.2)
    ax.fill_between(engine1['time_cycles'], engine1[sensor], alpha=0.15, color='steelblue')
    ax.set_title(sensor, fontsize=9, fontweight='bold')
    ax.set_xlabel('Cycles', fontsize=7)
    ax.grid(True, alpha=0.3)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Sensor Degradation Curves — Engine #1', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_sensor_degradation.png'), dpi=120)
plt.close()
print("03_sensor_degradation.png")

# ─── 4. Correlation Heatmap — Sensors vs RUL ──────────────────────────────────
corr = train[useful_sensors + ['RUL']].corr()['RUL'].drop('RUL').sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
colors_corr = ['tomato' if v < 0 else 'steelblue' for v in corr.values]
ax.barh(corr.index, corr.values, color=colors_corr)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson Correlation with RUL')
ax.set_title('Sensor Correlation with Remaining Useful Life')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '04_rul_correlation.png'), dpi=120)
plt.close()
print("04_rul_correlation.png")

# ─── 5. Full Sensor Correlation Heatmap ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = train[useful_sensors].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 7})
ax.set_title('Sensor Inter-Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '05_sensor_heatmap.png'), dpi=120)
plt.close()
print("05_sensor_heatmap.png")

# ─── 6. PCA of Sensor Space — colored by RUL ─────────────────────────────────
sample = train[useful_sensors + ['RUL']].sample(min(5000, len(train)), random_state=42)
X_pca = StandardScaler().fit_transform(sample[useful_sensors])
pca = PCA(n_components=2)
components = pca.fit_transform(X_pca)

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(components[:, 0], components[:, 1],
                c=sample['RUL'], cmap='RdYlGn', alpha=0.5, s=8)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('RUL (cycles)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA of Sensor Space — Colored by RUL', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '06_pca_sensors.png'), dpi=120)
plt.close()
print(f"06_pca_sensors.png  |  PCA variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# ─── 7. RUL distribution by degradation stage ─────────────────────────────────
train['stage'] = pd.cut(train['RUL'],
                         bins=[-1, 30, 80, 150, train['RUL'].max()+1],
                         labels=['Critical (<30)', 'Warning (30-80)', 'Caution (80-150)', 'Healthy (>150)'])
stage_counts = train['stage'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].pie(stage_counts, labels=stage_counts.index, autopct='%1.1f%%',
            colors=['tomato', 'orange', 'gold', 'mediumseagreen'], startangle=90)
axes[0].set_title('Engine-Cycle Distribution by Degradation Stage')

train.groupby('stage', observed=False)['RUL'].mean().plot(kind='bar', ax=axes[1],
    color=['tomato', 'orange', 'gold', 'mediumseagreen'], edgecolor='white')
axes[1].set_ylabel('Mean RUL (cycles)')
axes[1].set_title('Mean RUL per Stage')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '07_rul_stages.png'), dpi=120)
plt.close()
print("07_rul_stages.png")

# ─── 8. Multi-engine RUL countdown ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
palette = plt.cm.tab10.colors
for idx, uid in enumerate(range(1, 11)):
    eng = train[train['unit_id'] == uid].sort_values('time_cycles')
    ax.plot(eng['time_cycles'], eng['RUL'], color=palette[idx % 10], linewidth=1.2, label=f'Engine {uid}')
ax.axhspan(0, 30, alpha=0.15, color='tomato', label='Critical zone')
ax.axhspan(30, 80, alpha=0.1, color='orange', label='Warning zone')
ax.set_xlabel('Time (cycles)')
ax.set_ylabel('RUL (cycles)')
ax.set_title('RUL Countdown — First 10 Engines', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_rul_countdown.png'), dpi=120)
plt.close()
print("08_rul_countdown.png")

print(f"\n{'='*55}")
print(f"  EDA complete — {len(os.listdir(OUTPUT_PATH))} figures saved to:")
print(f"  {OUTPUT_PATH}")
print(f"{'='*55}")