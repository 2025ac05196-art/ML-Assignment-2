# Classification Model Comparison Streamlit App

## 1. Problem Statement

The objective of this project is to build an interactive machine learning web application that compares multiple classification algorithms on the same dataset. The application allows users to upload a CSV dataset, select the target column, choose a classification model, and view evaluation results such as accuracy, AUC, precision, recall, F1 score, MCC score, confusion matrix, and classification report.

This project is prepared for Machine Learning Assignment 2 as part of the M.Tech AIML/DSE Work Integrated Learning Programme.

## 2. Dataset Description

For the default demonstration, this project uses the Wisconsin Diagnostic Breast Cancer dataset, which is available through `scikit-learn` and based on the UCI Machine Learning Repository dataset.

- Dataset type: Classification
- Classification type: Binary classification
- Number of instances: 569
- Number of input features: 30
- Target column: `diagnosis`
- Target classes: `benign`, `malignant`

The dataset satisfies the assignment requirement of at least 500 instances and at least 12 features.

Users can also upload their own CSV classification dataset through the Streamlit app. The uploaded dataset should contain one target column and multiple feature columns.

## 3. GitHub Repository Link

Replace this placeholder with your GitHub repository link after uploading the project:

```text
https://github.com/your-username/your-repository-name
```

## 4. Live Streamlit App Link

Replace this placeholder with your deployed Streamlit Community Cloud link:

```text
https://your-app-name.streamlit.app/
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
| Logistic Regression | Usually performs well when the decision boundary is approximately linear. It is fast, interpretable, and effective for the default breast cancer dataset after feature scaling. |
| Decision Tree | Easy to interpret but may overfit if the tree becomes too deep. Performance can vary depending on the train-test split. |
| K-Nearest Neighbors | Sensitive to feature scaling and the value of K. It may perform well on clean datasets but can become slower for large datasets. |
| Naive Bayes | Fast and simple. It assumes conditional independence between features, so performance may be lower when features are strongly correlated. |
| Random Forest | Usually gives strong performance because it combines multiple decision trees and reduces overfitting compared with a single tree. |
| Overall Winner for This Dataset | Update this after reviewing the metric table. The best model is usually the one with the highest F1, AUC, and MCC scores. |

## 9. Streamlit App Features

The Streamlit app includes the following required features:

- CSV dataset upload option
- Target column selection
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix display
- Classification report display
- Default dataset fallback when no CSV file is uploaded

## 10. Project Structure

```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
```

Note: If you save trained model files separately, place them inside the `model/` folder.

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
