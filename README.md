# Classification Model Comparison Streamlit App

## 1. Problem Statement

The objective of this project is to build an interactive machine learning web application that compares multiple classification algorithms on the same dataset. The application allows users to upload a CSV dataset, select the target column, choose a classification model, and view evaluation results such as accuracy, AUC, precision, recall, F1 score, MCC score, confusion matrix, and classification report.

This project is prepared for Machine Learning Assignment 2 as part of the M.Tech AIML/DSE Work Integrated Learning Programme.

## 2. Dataset Description

This project uses the NPHA doctor visits classification dataset.

- Dataset type: Classification
- File name: `NPHA-doctor-visits.csv`
- Target column: Update this based on the final target column used in your notebook/app
- Minimum requirement: The dataset should contain at least 500 instances and 12 features

Users can also upload a CSV classification dataset through the Streamlit app. The uploaded dataset should contain one target column and multiple feature columns.

## 3. GitHub Repository Link

```text
https://github.com/2025ac05196-art/ML-Assignment-2
```

## 4. Live Streamlit App Link

```text
https://fsiwefw4bfvmrtc82f3yjp.streamlit.app/
```

## 5. Models Used

The following classification models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier

## 6. Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient, also called MCC Score

## 7. Model Comparison Table

After running the Streamlit application, copy the calculated metric values from the app and paste them into the table below.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Add result | Add result | Add result | Add result | Add result | Add result |
| Decision Tree | Add result | Add result | Add result | Add result | Add result | Add result |
| K-Nearest Neighbors | Add result | Add result | Add result | Add result | Add result | Add result |
| Naive Bayes | Add result | Add result | Add result | Add result | Add result | Add result |
| Random Forest | Add result | Add result | Add result | Add result | Add result | Add result |

## 8. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Performs well when the relationship between features and target is approximately linear. It is fast and interpretable. |
| Decision Tree | Easy to interpret but may overfit if the tree becomes too complex. |
| K-Nearest Neighbors | Sensitive to feature scaling and the value of K. Works well when similar records belong to similar classes. |
| Naive Bayes | Fast and simple. It assumes feature independence, so performance may reduce when features are highly correlated. |
| Random Forest | Generally provides strong results because it combines multiple decision trees and reduces overfitting. |
| Overall Winner for This Dataset | Update this after checking the metric table. The best model should have strong F1, AUC, and MCC values. |

## 9. Streamlit App Features

The Streamlit app includes the following required features:

- CSV dataset upload option
- Target column selection
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix display
- Classification report display

## 10. Project Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- NPHA-doctor-visits.csv
|-- train_models.ipynb
```

If a separate test dataset is created, keep it as:

```text
test_data.csv
```

## 11. How to Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## 12. Deployment Steps on Streamlit Community Cloud

1. Push the project folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Sign in using your GitHub account.
4. Click **New app**.
5. Select your repository and branch.
6. Select `app.py` as the main file.
7. Click **Deploy**.
8. Copy the live app URL and paste it in this README and in the final PDF submission.

## 13. Final Submission Checklist

- GitHub repository link is working.
- Streamlit app link opens correctly.
- App loads without errors.
- Dataset upload option works.
- Model dropdown works.
- Evaluation metrics are displayed.
- Confusion matrix or classification report is displayed.
- README content is added to the final PDF.
- Screenshot from BITS Virtual Lab is included in the final PDF.
