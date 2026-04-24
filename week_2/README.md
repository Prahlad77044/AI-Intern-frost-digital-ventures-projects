# Customer Churn Prediction

Production-ready, cost-sensitive customer churn prediction pipeline for a telecom company using the public IBM Telco Customer Churn dataset.

## Quick Links

- Dataset: `telco-customer-churn-by-IBM.csv`  
- Model file: `final_churn_model_calibrated_isotonic.pkl`  
- API: FastAPI endpoint at `/predict`  
- Threshold: **0.1275** (cost-minimizing)

---

## Business Costs & Optimal Threshold

| Prediction \ True | Churn = Yes (1) | Churn = No (0) |
|-------------------|-----------------|----------------|
| Predicted Yes | Correct | False Positive → **$1** (retention offer) |
| Predicted No | False Negative → **$5** (lost revenue) | Correct |

**Optimal threshold**: **0.1275** (minimizes total cost = 5×FN + 1×FP)

---

## Model Card

| Item | Details |
|------|---------|
| Model | XGBoost Classifier |
| Calibration | Isotonic Regression |
| Key Hyperparameters | `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`, `scale_pos_weight=1` |
| Threshold | **0.1275** |
| F1 Score | **0.742** |
| PR-AUC | **0.820** |
| Final Model | `final_churn_model_calibrated_isotonic.pkl` |

### Fairness & Slice Monitoring

Metrics tracked on: **tenure groups**, **Contract**, **PaymentMethod**  
Small subgroups may have high variance.

---

## Dataset Summary

- Rows: 7,043  
- Columns: 21  
- Target: `Churn` (Yes/No → 1/0)  
- MD5 checksum: `0f9de68e012bd3aed5fa7cdc9fc421af`

### Columns

- **ID**: `customerID` (dropped)  
- **Numerical**: `SeniorCitizen`, `tenure`, `MonthlyCharges`  
- **Categorical**: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `TotalCharges` (dropped), `Churn`

---

## Data Leakage & Feature Decisions

| Feature | Status | Reason |
|---------|--------|--------|
| customerID | Dropped | Unique identifier |
| TotalCharges | Dropped | High correlation with tenure × MonthlyCharges |
| All other features | Kept | No leakage, available at prediction time |

---

## Usage

### 1. Load & Predict (Python)

```python
import joblib
import pandas as pd

pipeline = joblib.load("final_churn_model_calibrated_isotonic.pkl")
threshold = 0.1275

# X_new: DataFrame with same columns as training (no customerID/Churn)
proba = pipeline.predict_proba(X_new)[:, 1]
pred  = (proba >= threshold).astype(int)
```

### 2. Run API

```bash
uvicorn api:app --reload
```

POST → `http://127.0.0.1:8000/predict`

### 3. Train / Explore

```bash
jupyter notebook
```

---

## Setup & Environment

**Python version**: 3.11.0

### Install

```bash
python -m venv churn_env

# Windows
.\churn_env\Scripts\activate

# macOS / Linux
source churn_env/bin/activate

pip install -r requirements.txt
```

### Key Dependencies

```text
fastapi==0.121.3
uvicorn==0.38.0
xgboot==3.1.2
scikit-learn==1.7.2
shap==0.50.0
pandas==2.3.3
joblib==1.5.2
mlflow==3.6.0
matplotlib==3.10.7
seaborn==0.13.2
```

---

## Decision Log

- Dropped `TotalCharges` → collinear  
- Chose Isotonic Regression → superior calibration  
- Threshold 0.1275 → minimizes expected dollar loss  
- Added slice metrics → early bias detection

## Monitoring Recommendations

- Log predictions + outcomes  
- Track calibration drift  
- Monitor slice performance  
- Retrain on significant drift