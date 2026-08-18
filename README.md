# Classification Model Comparison Streamlit App

## 1. Problem Statement

This project implements and compares five classification algorithms on the same public dataset and provides an interactive Streamlit application for evaluating the saved models on held-out test data. The application reports Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC), a confusion matrix, and a classification report.

The workflow covers dataset preparation, leakage-safe preprocessing, model training, evaluation, model serialization, UI development, and deployment with Streamlit Community Cloud.

## 2. Dataset Description

This project uses the **National Poll on Healthy Aging (NPHA)** dataset from the UCI Machine Learning Repository.

- **Dataset source:** [UCI Machine Learning Repository — National Poll on Healthy Aging](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+%28npha%29)
- **Dataset type:** Tabular, multi-class classification
- **Instances:** 714
- **Predictor features:** 14
- **Target column:** `Number of Doctors Visited`
- **Target classes:**
  - `1`: 0–1 doctors
  - `2`: 2–3 doctors
  - `3`: 4 or more doctors

### Predictor columns used

1. `Age`
2. `Phyiscal Health`
3. `Mental Health`
4. `Dental Health`
5. `Employment`
6. `Stress Keeps Patient from Sleeping`
7. `Medication Keeps Patient from Sleeping`
8. `Pain Keeps Patient from Sleeping`
9. `Bathroom Needs Keeps Patient from Sleeping`
10. `Uknown Keeps Patient from Sleeping`
11. `Trouble Sleeping`
12. `Prescription Sleep Medication`
13. `Race`
14. `Gender`

The spellings above match the source CSV column names.

### Preprocessing

- Empty rows and empty columns are removed.
- Refused-response values encoded as `-1` in predictor columns are converted to missing values.
- The data is split once using an **80:20 stratified train-test split** with `random_state=42`.
- Preprocessing is fitted only on the training partition.
- Numerical values are imputed with the training median and standardized.
- Categorical values are imputed with the most frequent training value and one-hot encoded.
- The target column is excluded from all preprocessing and predictor matrices.
- The held-out partition is saved as `test_data.csv` and is not used to fit the models.

## 3. Project Links

- **GitHub Repository:** [ML Assignment 2 Repository](https://github.com/2025ac05196-art/ML-Assignment-2)
- **Live Streamlit Application:** [Open the Streamlit App](https://fsiwefw4bfvmrtc82f3yjp.streamlit.app/)

## 4. Models Used

1. **Logistic Regression** — linear probabilistic baseline for multi-class classification.
2. **Decision Tree Classifier** — non-parametric tree model.
3. **K-Nearest Neighbors Classifier** — distance-based instance classifier.
4. **Gaussian Naive Bayes Classifier** — probabilistic classifier based on conditional independence.
5. **Random Forest Classifier** — ensemble of decision trees.

## 5. Evaluation Metrics

Every model is evaluated on the same held-out test partition using:

- Accuracy
- Multi-class AUC using weighted One-vs-Rest
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Matthews Correlation Coefficient (MCC)

## 6. Model Comparison Results

The following values are the previously recorded results. Re-run `save_models.py` and update this table if the dataset or package versions change.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5524 | 0.6161 | 0.4672 | 0.5524 | 0.4487 | 0.1547 |
| Decision Tree | 0.3566 | 0.4630 | 0.3629 | 0.3566 | 0.3590 | -0.0585 |
| K-Nearest Neighbors | 0.4965 | 0.5399 | 0.4854 | 0.4965 | 0.4683 | 0.1083 |
| Naive Bayes | 0.2937 | 0.6217 | 0.4633 | 0.2937 | 0.2510 | 0.1427 |
| Random Forest | 0.5035 | 0.5600 | 0.4587 | 0.5035 | 0.4593 | 0.0942 |

## 7. Performance Observations

### Logistic Regression

Logistic Regression achieved the highest recorded accuracy, recall, and MCC. Its AUC indicates moderate class-discrimination ability. It is selected as the overall model because it provides the strongest balance under the primary criteria and is computationally efficient and interpretable.

### Decision Tree

The Decision Tree produced the lowest AUC and a negative MCC, suggesting weak generalization on the held-out split. Depth control, minimum leaf size, pruning, and cross-validated hyperparameter tuning may improve performance.

### K-Nearest Neighbors

KNN achieved the highest recorded precision and F1 score. Because it is sensitive to feature scale, all transformed numerical inputs are standardized. Tuning the number of neighbors, distance measure, and weighting strategy may improve results.

### Naive Bayes

Naive Bayes achieved the highest recorded AUC but low accuracy and recall. Its probability ranking was more useful than its final class assignments. Correlated health attributes may weaken the model's conditional-independence assumption.

### Random Forest

Random Forest performed better than the single Decision Tree, indicating that the ensemble improved stability. Its MCC remained limited; tuning tree depth, feature sampling, class weights, and the number of estimators may help.

### Overall Winner

**Logistic Regression** is selected as the overall winner because it achieved the highest recorded accuracy, recall, and MCC. KNN achieved the highest precision and F1 score, while Naive Bayes achieved the highest AUC; therefore, no model dominated every metric.

## 8. Streamlit Application Features

- Use the generated held-out `test_data.csv` or upload another compatible test CSV.
- Select one of five saved models.
- Compare all models on the same uploaded test dataset.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display a confusion matrix and classification report for the selected model.
- Download the comparison metrics as CSV.
- Validate required columns and reject incompatible files.

## 9. Repository Structure

```text
ML-Assignment-2/
├── app.py
├── save_models.py
├── train_models.ipynb
├── requirements.txt
├── README.md
├── NPHA-doctor-visits.csv
├── test_data.csv
├── model_comparison_results.csv
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── metadata.joblib
    └── index.json
```

## 10. Running Locally

```bash
git clone https://github.com/2025ac05196-art/ML-Assignment-2.git
cd ML-Assignment-2
python -m pip install -r requirements.txt
python save_models.py
streamlit run app.py
```

`save_models.py` expects `NPHA-doctor-visits.csv` in the project root. It creates the held-out `test_data.csv`, comparison CSV, and serialized model files.

## 11. Deployment on Streamlit Community Cloud

1. Push the complete project to a public GitHub repository.
2. Ensure the generated `models/` files and `test_data.csv` are committed.
3. Open Streamlit Community Cloud.
4. Create a new app from the repository.
5. Select the `main` branch and `app.py`.
6. Deploy and verify that the application loads without errors.

## 12. Assignment Compliance Summary

- Public classification dataset with at least 500 instances
- At least 12 predictor features
- Five required classification models implemented on the same dataset
- Six required evaluation metrics reported
- Model comparison and model-wise observations included
- CSV test-data option and model-selection control included
- Confusion matrix and classification report included
- GitHub and live Streamlit links included
- Saved model files included
- Held-out test partition is not used for fitting

## 13. Final Submission Note

The final submission PDF should contain these items in the required order:

1. Clickable GitHub repository link
2. Clickable live Streamlit application link
3. One readable screenshot showing execution on the BITS Virtual Lab
4. The complete README content

Before submission, verify that the repository is public, the deployed application opens correctly, all generated model files are present, both links are clickable, and the BITS Virtual Lab screenshot is readable.
