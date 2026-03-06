# Churn Sentinel — Model Evaluation Report


## Baseline: Logistic Regression

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 73.24% |
| Precision | 49.75% |
| Recall    | 79.41% |
| F1-Score  | 61.17% |
| ROC-AUC   | 84.76% |

**Figures:**
- `reports/figures/06_cm_baseline.png`
- `reports/figures/07_roc_baseline.png`
- `reports/figures/08_coef_baseline.png`

---

## Improved Model: XGBoost

### Best Hyperparameters
```
  colsample_bytree: 0.8
  learning_rate: 0.05
  max_depth: 3
  n_estimators: 200
  subsample: 0.8
```

### Results

| Metric    | Logistic Regression | XGBoost | Improvement |
|-----------|--------------------:|--------:|------------:|
| Accuracy  | 73.24% | 75.23% | +1.99% |
| Precision | 49.75% | 52.16% | +2.41% |
| Recall    | 79.41% | 80.75% | +1.34% |
| F1-Score  | 61.17% | 63.38% | +2.21% |
| ROC-AUC   | 84.76% | 84.75% | -0.01% |

### Figures
- `reports/figures/09_cm_xgboost.png`
- `reports/figures/10_roc_comparison.png`
- `reports/figures/11_feature_importance_xgb.png`
- `reports/figures/12_metrics_comparison.png`

---
