# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease using Logistic Regression, with a focus on medical-context evaluation metrics and threshold tuning.

## Overview

This project applies binary classification to predict whether a patient has heart disease based on clinical features, using the Heart Failure Prediction dataset from Kaggle.

## Dataset

- Rows: 918 patients
- Features: 11 clinical indicators (age, cholesterol, resting blood pressure, chest pain type, etc.)
- Target: HeartDisease (1 = disease present, 0 = no disease)
- Class balance: 55.3% disease, 44.7% no disease (reasonably balanced)

## Workflow

1. Exploratory Data Analysis
   - Checked class balance before modeling, since imbalanced classes can make accuracy misleading
   - Explored correlations between features and target; ST_Slope_Flat showed the strongest positive correlation with heart disease, consistent with known cardiology patterns

2. Feature Encoding
   - Binary columns (Sex, ExerciseAngina) mapped directly to 0/1
   - Nominal columns with no clear ordinal relationship (ChestPainType, RestingECG, ST_Slope) encoded using One-Hot Encoding

3. Feature Scaling
   - Applied StandardScaler, fit only on training data to avoid data leakage

4. Modeling
   - Trained a Logistic Regression model
   - Evaluated using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC

5. Threshold Tuning (Key Insight)
   - Default threshold (0.5): Recall = 0.84, Precision = 0.90
   - Lowered threshold to 0.3: Recall improved to 0.90 with minimal precision trade-off (0.89)
   - In a medical context, missing a disease case (false negative) is more costly than a false alarm, so prioritizing recall is the appropriate choice

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.85 |
| Precision (Disease) | 0.90 |
| Recall (Disease) | 0.84 (0.90 after threshold tuning) |
| F1-Score (Disease) | 0.87 |
| ROC-AUC | 0.927 |

## Key Takeaways

- A high accuracy score alone doesn't guarantee a good model in imbalanced or high-stakes contexts; precision, recall, and AUC give a fuller picture
- Threshold tuning allows adapting a model's behavior to the cost of different error types without retraining
- Feature correlations aligned with established medical knowledge (ST_Slope_Flat), adding confidence in the model's learned patterns

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (LogisticRegression, StandardScaler, train_test_split, confusion_matrix, roc_curve)
- Matplotlib, Seaborn

## Files

- Heart_disease_prediction.ipynb: full analysis and modeling notebook

## How to Run

1. Clone this repository
2. Open Heart_disease_prediction.ipynb in Jupyter Notebook or Google Colab
3. Download the dataset from Kaggle (Heart Failure Prediction Dataset) and place heart.csv in the same directory
4. Run all cells
