# Customer Transaction Prediction

A binary classification project that predicts whether a customer will make a specific transaction in the future, using 200 anonymized numeric features. The notebook covers data-quality checks, statistical EDA, model comparison, threshold tuning and a saved final model.

## Problem Statement

Given anonymized customer data (`var_0` … `var_199`), predict the binary `target` (1 = customer made the transaction, 0 = did not). Because the features are anonymized, business-driven EDA is not possible, so the analysis focuses on statistics: data quality, target balance, distributions, feature-target signal and feature-feature dependence.

## Dataset

| Item | Details |
|---|---|
| Rows | 200,000 |
| Columns | 202 (`ID_code`, `target`, `var_0` … `var_199`) |
| Features | 200 anonymized numeric (float64) features |
| Target | Binary, imbalanced (about 90% class 0 / 10% class 1) |
| Source | Santander Customer Transaction Prediction (Kaggle) |

The data file is not included in this repository. Download `train.csv` from Kaggle and place it in the project folder (the notebook reads it as `train(1).csv`, so rename it or change the path in the load cell).

## Key Findings from EDA

- No missing values, no duplicate rows, and all `ID_code` values are unique.
- Target is imbalanced: 89.95% class 0 and 10.05% class 1.
- Features are close to symmetric (mean absolute skew about 0.08).
- Each feature has only weak linear correlation with the target (strongest is about -0.08, for `var_81`).
- Features are almost independent of each other (maximum absolute correlation of 0.0098, no pair above 0.1).

## Approach

1. **Setup and loading**: load data, separate features from `ID_code` and `target`.
2. **Data analysis report**: data quality, target distribution, descriptive stats, skew and cardinality, IQR outliers, feature-vs-target signal, feature-feature dependence.
3. **Split**: stratified 70% train / 15% validation / 15% test (same positive rate of 10.05% in each). Validation is used for model selection and early stopping, and the test set is used only once at the end.
4. **Models compared**: Logistic Regression (scaled, class-weighted), Gaussian Naive Bayes, Random Forest (class-weighted) and LightGBM (strongly regularized, early stopping on validation AUC). Scaling sits inside pipelines to avoid data leakage.
5. **Selection and threshold**: choose the model with the best validation ROC-AUC, then pick the F1-optimal decision threshold on the validation set.
6. **Final evaluation and export**: evaluate once on the test set, plot feature importance and save the model.

## Results

### Model comparison (validation set)

| Model | ROC-AUC | PR-AUC | Fit time (s) |
|---|---|---|---|
| **LightGBM** | **0.8959** | **0.5925** | 13.9 |
| Gaussian Naive Bayes | 0.8861 | 0.5723 | 0.6 |
| Logistic Regression | 0.8566 | 0.5000 | 1.0 |
| Random Forest | 0.8201 | 0.4049 | 39.0 |

### Final model on the held-out test set (LightGBM, threshold = 0.2647)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8959 |
| PR-AUC | 0.6046 |
| Accuracy | 0.9085 |
| Precision (class 1) | 0.5429 |
| Recall (class 1) | 0.5647 |
| F1 (class 1) | 0.5536 |

Because the target is imbalanced, ROC-AUC and PR-AUC are the main metrics, and accuracy alone is not a reliable measure.

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- LightGBM
- Joblib
- Sweetviz (automated EDA report)

## Project Structure

```
├── Customer_Transaction__Prediction.ipynb   # Main notebook
├── train.csv                                # Dataset (download from Kaggle, not included)
├── best_model.pkl                           # Saved model, threshold and feature list (generated)
├── submission.csv                           # Predictions for test.csv (generated, optional)
├── SWEETVIZ_REPORT.html                     # Automated EDA report (generated)
└── README.md
```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/A-praveen123/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn lightgbm joblib sweetviz jupyter

# 3. Add the dataset (train.csv) to the project folder, then launch
jupyter notebook Customer_Transaction__Prediction.ipynb
```

Run the cells from top to bottom. If `test.csv` is present, the last cells also create a `submission.csv` file.

## Using the Saved Model

```python
import joblib
import pandas as pd

artifact = joblib.load("best_model.pkl")
model, threshold, feats = artifact["model"], artifact["threshold"], artifact["features"]

new_data = pd.read_csv("new_customers.csv")
proba = model.predict_proba(new_data[feats])[:, 1]
prediction = (proba >= threshold).astype(int)
```

## Future Improvements

- Tune LightGBM hyperparameters with Optuna or cross-validation.
- Add feature engineering based on value counts and frequency encoding, which often help on this kind of anonymized data.
- Try ensembling LightGBM with Naive Bayes.
- Use stratified k-fold cross-validation for more stable estimates.

## Author

**A. Praveen**
Aspiring Data Scientist | B.Tech in Computer Science

- LinkedIn: [praveen-a-21b36531a](https://linkedin.com/in/praveen-a-21b36531a)
- GitHub: [A-praveen123](https://github.com/A-praveen123)
