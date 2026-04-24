model_card_md = """
# Model Card: XGBoost Churn Classifier

## Model Overview
- **Name:** XGBoost Churn Classifier
- **Version:** 1.0
- **Author:** Prahlad Neupane
- **Description:** A machine learning pipeline consisting of a preprocessing step followed by an XGBoost classifier, trained for customer churn prediction. The model has been calibrated using **Isotonic Regression** to produce reliable probability estimates.

## Performance Metrics (on test set)
| Metric      | Value |
|------------|-------|
| Overall F1 | {0.6049} |
| Overall PR-AUC | {0.6456} |

### Slice Metrics
Metrics were computed for key features to understand performance across different groups.  

**Tenure Bins:** 0-12, 13-24, 25-36, 37-48, 49-60, 60+  
**Contract Types:** Month-to-month, One year, Two year  
**Payment Methods:** Bank transfer, Credit card, Electronic check, Mailed check  

*Note: Small groups may have unstable metrics.*

## Fairness Notes
- The model shows variations in performance for short-tenure customers and certain payment methods.
- Some minor groups may have less reliable predictions due to limited data.

## Hyperparameters
- **learning_rate:** 0.1
- **max_depth:** 3
- **n_estimators:** 100
- **scale_pos_weight:** 1
- **Other XGBoost defaults** apply.

## Calibration
- **Method:** Isotonic Regression
- **Purpose:** To provide well-calibrated probability estimates, especially for downstream threshold-based decision-making.

## Assumptions
- Input data should match the same preprocessing schema as used in training.
- Categorical features must have the same levels as seen during training.
- Numeric features should be in the same range as the training dataset.

## Limitations
- Performance may drop on unseen distributions.
- Small slice groups can have unstable F1 or PR-AUC.
- Biases detected in some payment methods or short-tenure customers.

## Usage
- Use `predict_proba` to get probability of churn.
- Use `predict` to get class labels based on a threshold (e.g., 0.1275 for cost-minimizing threshold).
- Probabilities are calibrated and should be used for business threshold decisions.

## Example Usage (Python)
```python
import joblib
pipeline = joblib.load("final_churn_model_calibrated_isotonic.pkl")
threshold = 0.1275

# Predict on new customer data
y_prob = pipeline.predict_proba(X_new)[:,1]
y_pred = (y_prob >= threshold).astype(int)
