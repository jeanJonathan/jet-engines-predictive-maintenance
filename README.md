# Predictive Maintenance using NASA C-MAPSS Dataset

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. It includes data processing, model training, and basic analysis related to predictive maintenance.

## Goal

* Load and explore the C-MAPSS dataset
* Predict engine Remaining Useful Life (RUL)
* Prepare a basic workflow for predictive maintenance models

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run data exploration:

```
python src/01_data_exploration.py
```

Run preprocessing and training:

```
python src/02_preprocessing.py
python src/03_train_model.py
```

Run dashboard:
```
python src/04_dashboard.py
```

Run cost analysis:
```
python src/05_cost_analysis.py
```

## Dataset Reference

The dataset is based on the NASA C-MAPSS engine simulation.

**Reference:**

[1] A. Saxena, K. Goebel, D. Simon and N. Eklund,
"Damage propagation modeling for aircraft engine run-to-failure simulation,"
2008 International Conference on Prognostics and Health Management,
Denver, CO, USA, 2008, pp. 1–9, doi: 10.1109/PHM.2008.4711414.

