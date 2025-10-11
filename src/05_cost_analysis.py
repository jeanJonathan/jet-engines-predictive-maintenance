import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_PATH = os.path.join(DATA_PATH, 'reports')
os.makedirs(OUTPUT_PATH, exist_ok=True)

model = keras.models.load_model(os.path.join(DATA_PATH, 'models', 'lstm_model.keras'))
X_test = np.load(os.path.join(DATA_PATH, 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'processed', 'y_test.npy'))
y_pred = model.predict(X_test).flatten()

UNSCHEDULED_MAINTENANCE_COST = 50000  # Emergency repair
SCHEDULED_MAINTENANCE_COST = 10000  # Planned maintenance
FLIGHT_HOUR_VALUE = 20000  # Mission capability lost per hour
TRADITIONAL_INTERVAL = 150
INSPECTION_WINDOW = 30 

def calculate_traditional_costs(actual_rul_list):
    total_cost = 0
    unscheduled = 0
    scheduled = 0

    for actual_rul in actual_rul_list:
        if actual_rul < INSPECTION_WINDOW:
            # RUL too low - fails before next scheduled inspection
            total_cost += UNSCHEDULED_MAINTENANCE_COST
            unscheduled += 1
        else:
            # Caught by scheduled inspection (may be premature)
            total_cost += SCHEDULED_MAINTENANCE_COST
            scheduled += 1

    return total_cost, scheduled, unscheduled

def calculate_predictive_costs(actual_rul_list, predicted_rul_list, threshold=50):
    total_cost = 0
    scheduled = 0
    unscheduled = 0

    for actual, predicted in zip(actual_rul_list, predicted_rul_list):
        if predicted < threshold:
            # AI recommends maintenance
            # Schedule it regardless (model triggered action)
            total_cost += SCHEDULED_MAINTENANCE_COST
            scheduled += 1
        else:
            # AI says engine is healthy - no maintenance scheduled
            if actual < threshold:
                # FALSE NEGATIVE: Missed a failing engine = emergency failure
                total_cost += UNSCHEDULED_MAINTENANCE_COST
                unscheduled += 1
            # else: TRUE NEGATIVE - correctly identified as healthy, no cost

    return total_cost, scheduled, unscheduled

trad_cost, trad_sched, trad_unsched = calculate_traditional_costs(y_test)
pred_cost, pred_sched, pred_unsched = calculate_predictive_costs(y_test, y_pred)

savings = trad_cost - pred_cost
savings_pct = (savings / trad_cost) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cost comparison
axes[0, 0].bar(['Traditional', 'Predictive'], [trad_cost, pred_cost], color=['red', 'green'])
axes[0, 0].set_ylabel('Total Cost ($)')
axes[0, 0].set_title('Maintenance Cost Comparison')
axes[0, 0].text(0, trad_cost/2, f'${trad_cost:,}', ha='center', fontsize=12, fontweight='bold')
axes[0, 0].text(1, pred_cost/2, f'${pred_cost:,}', ha='center', fontsize=12, fontweight='bold')

# Breakdown
categories = ['Scheduled', 'Unscheduled']
trad_events = [trad_sched, trad_unsched]
pred_events = [pred_sched, pred_unsched]

x = np.arange(len(categories))
width = 0.35

axes[0, 1].bar(x - width/2, trad_events, width, label='Traditional', color='red', alpha=0.7)
axes[0, 1].bar(x + width/2, pred_events, width, label='Predictive', color='green', alpha=0.7)
axes[0, 1].set_ylabel('Number of Events')
axes[0, 1].set_title('Maintenance Events Breakdown')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(categories)
axes[0, 1].legend()

# Savings breakdown
axes[1, 0].pie([savings, pred_cost], labels=[f'Savings\n${savings:,}', f'Remaining Cost\n${pred_cost:,}'],
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
axes[1, 0].set_title('Cost Savings Distribution')

# Mission readiness
readiness_trad = ((len(y_test) - trad_unsched) / len(y_test)) * 100
readiness_pred = ((len(y_test) - pred_unsched) / len(y_test)) * 100

axes[1, 1].bar(['Traditional', 'Predictive'], [readiness_trad, readiness_pred], color=['orange', 'green'])
axes[1, 1].set_ylabel('Operational Readiness (%)')
axes[1, 1].set_title('Fleet Availability')
axes[1, 1].set_ylim([0, 100])
axes[1, 1].axhline(y=90, color='r', linestyle='--', label='Target: 90%')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'cost_analysis.png'), dpi=150)

report_df = pd.DataFrame({
    'Metric': ['Total Cost', 'Scheduled Events', 'Unscheduled Events',
               'Operational Readiness (%)', 'Cost per Engine'],
    'Traditional': [f'${trad_cost:,}', trad_sched, trad_unsched,
                   f'{readiness_trad:.1f}', f'${trad_cost/len(y_test):,.0f}'],
    'Predictive': [f'${pred_cost:,}', pred_sched, pred_unsched,
                  f'{readiness_pred:.1f}', f'${pred_cost/len(y_test):,.0f}'],
    'Improvement': [f'${savings:,} ({savings_pct:.1f}%)',
                   f'{trad_sched - pred_sched}',
                   f'{trad_unsched - pred_unsched} ({((trad_unsched - pred_unsched)/max(trad_unsched,1)*100):.1f}%)',
                   f'{readiness_pred - readiness_trad:.1f}pp',
                   f'${(trad_cost - pred_cost)/len(y_test):,.0f}']
})

report_df.to_csv(os.path.join(OUTPUT_PATH, 'cost_report.csv'), index=False)

