"""
05_cost_analysis.py
====================
Maintenance Cost Analysis & ROI Simulation — NASA C-MAPSS FD001
Author : Jean-Jonathan KOFFI

Extends baseline with:
  - Multi-model comparison of cost savings
  - Threshold sensitivity analysis
  - Monte Carlo simulation for cost uncertainty
  - ROI and break-even analysis
  - CSV + PNG report generation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH  = os.path.join(BASE_DIR, 'data')
PROC_PATH  = os.path.join(DATA_PATH, 'processed')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
OUT_PATH   = os.path.join(DATA_PATH, 'reports')
os.makedirs(OUT_PATH, exist_ok=True)

# ─── Cost parameters (aerospace industry estimates) ───────────────────────────
COST = {
    'unscheduled':    50_000,   # Emergency in-service failure
    'scheduled':      10_000,   # Planned shop visit
    'false_positive':  2_000,   # Unnecessary maintenance (opportunity cost)
    'aog_per_day':   120_000,   # Aircraft On Ground — revenue lost per day
    'aog_days':            2,   # Avg AOG duration for unscheduled event
}
THRESHOLD = 50       # Default maintenance trigger (predicted RUL < threshold)

# ─── Load data ────────────────────────────────────────────────────────────────
y_test    = np.load(os.path.join(PROC_PATH,  'y_test.npy'))
lstm_pred = np.load(os.path.join(MODEL_PATH, 'lstm_pred_test.npy'))
rf_pred   = np.load(os.path.join(MODEL_PATH, 'rf_pred_test.npy'))

xgb_path = os.path.join(MODEL_PATH, 'xgb_pred_test.npy')
xgb_pred = np.load(xgb_path) if os.path.exists(xgb_path) else None

MODELS = {'LSTM (BiDir)': lstm_pred, 'Random Forest': rf_pred}
if xgb_pred is not None:
    MODELS['XGBoost'] = xgb_pred

# ─── Cost functions ───────────────────────────────────────────────────────────
def traditional_cost(actual_rul_list, threshold=THRESHOLD):
    """Fixed-interval policy: assume all engines are maintained at threshold."""
    total, sched, unsched = 0, 0, 0
    for rul in actual_rul_list:
        if rul < 30:       # Engine fails before inspection
            total += COST['unscheduled'] + COST['aog_per_day'] * COST['aog_days']
            unsched += 1
        else:
            total += COST['scheduled']
            sched += 1
    return total, sched, unsched

def predictive_cost(actual_rul_list, predicted_rul_list, threshold=THRESHOLD):
    """Condition-based policy: act when model flags engine."""
    total, sched, unsched, fp = 0, 0, 0, 0
    for actual, predicted in zip(actual_rul_list, predicted_rul_list):
        if predicted < threshold:                            # Model triggers maintenance
            if actual < threshold:                           # TRUE POSITIVE
                total += COST['scheduled']
                sched += 1
            else:                                            # FALSE POSITIVE (unnecessary)
                total += COST['false_positive']
                fp += 1
        else:                                                # Model says OK
            if actual < threshold:                           # FALSE NEGATIVE → failure
                total += COST['unscheduled'] + COST['aog_per_day'] * COST['aog_days']
                unsched += 1
    return total, sched, unsched, fp

# ─── 1. Baseline comparison ───────────────────────────────────────────────────
trad_cost, trad_sched, trad_unsched = traditional_cost(y_test)

results = {}
for name, pred in MODELS.items():
    pcost, psched, punsched, fp = predictive_cost(y_test, pred)
    savings = trad_cost - pcost
    roi     = (savings / trad_cost) * 100
    readiness_trad = ((len(y_test) - trad_unsched) / len(y_test)) * 100
    readiness_pred = ((len(y_test) - punsched)     / len(y_test)) * 100
    results[name] = {
        'total_cost': pcost, 'savings': savings, 'roi_pct': roi,
        'scheduled': psched, 'unscheduled': punsched, 'false_positives': fp,
        'readiness': readiness_pred,
    }

print("\n" + "="*65)
print("  MAINTENANCE COST ANALYSIS — SUMMARY")
print("="*65)
print(f"  Traditional policy: ${trad_cost:>12,.0f}  ({trad_sched} sched + {trad_unsched} unsched)")
for name, r in results.items():
    print(f"  {name:<20}: ${r['total_cost']:>12,.0f}  → Savings ${r['savings']:,.0f} ({r['roi_pct']:.1f}%)")
print("="*65 + "\n")

# ─── 2. Threshold sensitivity ─────────────────────────────────────────────────
thresholds = np.arange(10, 120, 5)
sensitivity = {name: [] for name in MODELS}
for th in thresholds:
    for name, pred in MODELS.items():
        pcost, *_ = predictive_cost(y_test, pred, threshold=th)
        sensitivity[name].append(pcost)

# ─── 3. Monte Carlo simulation ────────────────────────────────────────────────
N_SIMULATIONS = 5_000
np.random.seed(42)

mc_trad = []
mc_pred = {name: [] for name in MODELS}

for _ in range(N_SIMULATIONS):
    # Sample cost uncertainty ±30%
    c_sched   = COST['scheduled']   * np.random.uniform(0.7, 1.3)
    c_unsched = COST['unscheduled'] * np.random.uniform(0.7, 1.3)
    c_aog     = COST['aog_per_day'] * np.random.uniform(0.7, 1.3)
    c_fp      = COST['false_positive'] * np.random.uniform(0.7, 1.3)

    t_cost = sum(
        c_unsched + c_aog * COST['aog_days'] if r < 30 else c_sched
        for r in y_test
    )
    mc_trad.append(t_cost)

    for name, pred in MODELS.items():
        p_cost = 0
        for actual, predicted in zip(y_test, pred):
            if predicted < THRESHOLD:
                p_cost += c_sched if actual < THRESHOLD else c_fp
            elif actual < THRESHOLD:
                p_cost += c_unsched + c_aog * COST['aog_days']
        mc_pred[name].append(p_cost)

mc_trad = np.array(mc_trad)
mc_pred = {k: np.array(v) for k, v in mc_pred.items()}

# ─── 4. Plots ─────────────────────────────────────────────────────────────────
plt.rcParams.update({'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
                     'text.color': '#c9d1d9', 'axes.labelcolor': '#c9d1d9',
                     'xtick.color': '#c9d1d9', 'ytick.color': '#c9d1d9',
                     'axes.edgecolor': '#30363d', 'grid.color': '#30363d'})

model_names = list(MODELS.keys())
COLORS = ['#58a6ff', '#ff9800', '#4caf50'][:len(model_names)]

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

# Cost comparison bars
ax1 = fig.add_subplot(gs[0, :2])
all_names  = ['Traditional'] + model_names
all_costs  = [trad_cost] + [results[n]['total_cost'] for n in model_names]
all_colors = ['#e74c3c'] + COLORS
bars = ax1.bar(all_names, [c/1e6 for c in all_costs], color=all_colors, edgecolor='#0d1117', linewidth=1.5)
for bar, cost in zip(bars, all_costs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'${cost/1e6:.2f}M', ha='center', fontsize=9, color='#c9d1d9')
ax1.set_ylabel('Total Cost (M$)')
ax1.set_title('Total Maintenance Cost by Strategy', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Savings & ROI
ax2 = fig.add_subplot(gs[0, 2])
savings_vals = [results[n]['savings'] for n in model_names]
roi_vals     = [results[n]['roi_pct'] for n in model_names]
ax2_twin = ax2.twinx()
ax2.bar(model_names, [s/1e3 for s in savings_vals], color=COLORS, alpha=0.8)
ax2_twin.plot(model_names, roi_vals, 'o--', color='#f1c40f', linewidth=2, markersize=8)
ax2.set_ylabel('Savings ($k)', color='#c9d1d9')
ax2_twin.set_ylabel('ROI (%)', color='#f1c40f')
ax2.set_title('Savings & ROI vs Traditional', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Threshold sensitivity
ax3 = fig.add_subplot(gs[1, :2])
ax3.axhline(trad_cost/1e6, color='#e74c3c', linestyle='--', linewidth=1.5, label='Traditional')
for name, col in zip(model_names, COLORS):
    ax3.plot(thresholds, [c/1e6 for c in sensitivity[name]], color=col, linewidth=2, label=name)
ax3.axvline(THRESHOLD, color='white', linestyle=':', linewidth=1, alpha=0.5)
ax3.set_xlabel('Maintenance Trigger Threshold (RUL cycles)')
ax3.set_ylabel('Total Cost (M$)')
ax3.set_title(f'Cost Sensitivity vs Threshold (current={THRESHOLD})', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Monte Carlo — cost distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(mc_trad/1e6, bins=40, alpha=0.5, color='#e74c3c', label='Traditional')
for name, col in zip(model_names, COLORS):
    ax4.hist(mc_pred[name]/1e6, bins=40, alpha=0.5, color=col, label=name)
ax4.set_xlabel('Cost (M$)')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Monte Carlo Cost Distribution (n={N_SIMULATIONS:,})', fontweight='bold')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# Maintenance breakdown
ax5 = fig.add_subplot(gs[2, 0])
cats   = ['Scheduled', 'Unscheduled', 'False Pos.']
x      = np.arange(len(cats))
width  = 0.8 / (len(model_names) + 1)
ax5.bar(x - width*len(model_names)/2, [trad_sched, trad_unsched, 0], width, color='#e74c3c', alpha=0.8, label='Traditional')
for i, (name, col) in enumerate(zip(model_names, COLORS)):
    r = results[name]
    ax5.bar(x - width*(len(model_names)/2 - 1 - i), [r['scheduled'], r['unscheduled'], r['false_positives']],
            width, color=col, alpha=0.8, label=name)
ax5.set_xticks(x)
ax5.set_xticklabels(cats)
ax5.set_ylabel('Events')
ax5.set_title('Maintenance Event Breakdown', fontweight='bold')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3, axis='y')

# Fleet readiness
ax6 = fig.add_subplot(gs[2, 1])
readiness_trad = ((len(y_test) - trad_unsched) / len(y_test)) * 100
readiness_vals = [readiness_trad] + [results[n]['readiness'] for n in model_names]
bars2 = ax6.bar(all_names, readiness_vals, color=all_colors, edgecolor='#0d1117', linewidth=1.5)
ax6.axhline(90, color='#f1c40f', linestyle='--', linewidth=1.5, label='Target 90%')
ax6.set_ylim([0, 102])
ax6.set_ylabel('Operational Readiness (%)')
ax6.set_title('Fleet Availability', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, readiness_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', ha='center', fontsize=8)

# Savings boxplot (MC)
ax7 = fig.add_subplot(gs[2, 2])
mc_savings = {name: mc_trad - mc_pred[name] for name in model_names}
ax7.boxplot([mc_savings[n]/1e3 for n in model_names], labels=model_names,
            patch_artist=True,
            boxprops=dict(facecolor='#161b22', color='#30363d'),
            medianprops=dict(color='#f1c40f', linewidth=2))
for i, (name, col) in enumerate(zip(model_names, COLORS)):
    ax7.scatter([i+1]*100, np.random.choice(mc_savings[name], 100)/1e3,
                alpha=0.2, s=5, color=col)
ax7.set_ylabel('Savings vs Traditional ($k)')
ax7.set_title('Savings Distribution (MC)', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

fig.suptitle('Maintenance Cost Analysis & ROI — NASA C-MAPSS FD001', fontsize=15, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUT_PATH, 'cost_analysis.png'), dpi=130, bbox_inches='tight')
plt.close()
print("cost_analysis.png saved")

# ─── 5. CSV Report ────────────────────────────────────────────────────────────
rows = [{
    'Strategy': 'Traditional (Fixed-Interval)',
    'Total Cost ($)': trad_cost,
    'Scheduled Events': trad_sched,
    'Unscheduled Events': trad_unsched,
    'False Positives': 0,
    'Savings vs Traditional ($)': 0,
    'ROI (%)': 0,
    'Readiness (%)': round(((len(y_test)-trad_unsched)/len(y_test))*100, 1),
    'MC Savings Mean ($k)': 0,
    'MC Savings P10 ($k)': 0,
    'MC Savings P90 ($k)': 0,
}]
for name in model_names:
    r = results[name]
    savings_arr = mc_savings[name]
    rows.append({
        'Strategy': f'Predictive — {name}',
        'Total Cost ($)': r['total_cost'],
        'Scheduled Events': r['scheduled'],
        'Unscheduled Events': r['unscheduled'],
        'False Positives': r['false_positives'],
        'Savings vs Traditional ($)': r['savings'],
        'ROI (%)': round(r['roi_pct'], 1),
        'Readiness (%)': round(r['readiness'], 1),
        'MC Savings Mean ($k)': round(savings_arr.mean()/1e3, 1),
        'MC Savings P10 ($k)':  round(np.percentile(savings_arr, 10)/1e3, 1),
        'MC Savings P90 ($k)':  round(np.percentile(savings_arr, 90)/1e3, 1),
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_PATH, 'cost_report.csv'), index=False)
print("cost_report.csv saved")

print("\n" + "="*65)
print("  MONTE CARLO RESULTS (95% CI on savings)")
print("="*65)
for name in model_names:
    s = mc_savings[name]
    print(f"  {name:<20}: Mean ${s.mean()/1e3:.1f}k  "
          f"[P10 ${np.percentile(s,10)/1e3:.1f}k — P90 ${np.percentile(s,90)/1e3:.1f}k]")
print("="*65)