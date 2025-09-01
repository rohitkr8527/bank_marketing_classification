# Bank Marketing Classification – Kaggle Playground Series S5E8

This repository contains my solution for the **Kaggle Playground Series – Season 5, Episode 8** (Binary Classification with a Bank Dataset), where the task is to predict whether a customer subscribes to a term deposit.

## Competition Overview

This challenge is part of the **Playground Series**, designed for practicing ML on synthetic datasets generated from real-world data.

* **Competition Link**: [Kaggle Playground S5E8](https://www.kaggle.com/competitions/playground-series-s5e8)
* **Task**: Binary classification (term deposit subscription).

## Project Structure

```
├── model.ipynb        # Full Notebook: data preprocessing, modeling, evaluation
├── requirements.txt   # Python dependencies
├── submission.csv     # submission file
└── README.md          # Project documentation

```
## Dataset

* **Origin**: Synthetic data generated using deep learning from the original Bank Marketing dataset.
* **Features**: Demographics, financial details, and campaign outcomes.
* **Target**: `y` (1 = subscribed, 0 = not subscribed).

## Approach

1. **EDA**: Explored distributions and feature relationships.
2. **Preprocessing**: Encoded categorical variables, scaled numerical values.
3. **Modeling**: Used **XGBoost Classifier**.
4. **Evaluation**: ROC-AUC chosen as the main metric.

## Model Hyperparameters

```python
xgb_params = dict(
    colsample_bytree=0.8,
    gamma=0,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    eval_metric='logloss',  
    random_state=42
)
```

## Results

| Model Setup                  | ROC AUC    |
| ---------------------------- | ---------- |
| XGBoost (train/test split)   | 0.9658     |
| XGBoost (full training data) | **0.9706** |

## How to Run

```bash
git clone https://github.com/yourusername/bank-marketing-classification.git
cd bank-marketing-classification
pip install -r requirements.txt
jupyter notebook model.ipynb
```

## Future Improvements

* Add accuracy, precision, recall, F1-score.
* Try Random Forest, Logistic Regression, LightGBM.
* Feature selection with SHAP.
* Hyperparameter tuning with Optuna.
