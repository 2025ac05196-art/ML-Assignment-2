# Classification Model Comparison Streamlit App

## 1. Problem Statement

The objective of this project is to build an interactive machine learning web application that compares multiple classification algorithms on the same dataset. The application allows users to evaluate pre-trained classification models, select the target column, and view comprehensive evaluation metrics including accuracy, precision, recall, F1 score, AUC score, and Matthews Correlation Coefficient. This demonstrates the complete end-to-end ML deployment workflow: modeling, evaluation, UI design, and deployment on Streamlit Community Cloud.

## 2. Dataset Description

This project uses the NPHA  (National Poll on Healthy Aging) Doctor Visits classification dataset.

- **Dataset Type**: Multi-class Classification
- **Target Variable**: "Number of Doctors Visited" (Classification target)
- **Total Instances**: 714 samples (exceeds minimum requirement of 500)
- **Total Features**: 14 features (exceeds minimum requirement of 12)
- **Data Preprocessing**: Missing values (represented as -1) are handled by replacing with modal values
- **Train-Test Split**: 80-20 split with stratification to maintain class distribution
- **Feature Scaling**: StandardScaler applied for models requiring scaled features

Users can also upload their own CSV classification dataset through the Streamlit app. The uploaded dataset should contain one target column and multiple feature columns (minimum 12 features and 500 instances recommended).

## 3. GitHub Repository Link

```text
https://github.com/2025ac05196-art/ML-Assignment-2
```

## 4. Live Streamlit App Link

```text
https://fsiwefw4bfvmrtc82f3yjp.streamlit.app/
```

## 5. Models Used

The following five classification models are implemented and trained on the NPHA dataset:

1. **Logistic Regression**: A linear model for binary and multi-class classification
2. **Decision Tree Classifier**: A non-parametric model with recursive partitioning
3. **K-Nearest Neighbors (KNN) Classifier**: An instance-based learning algorithm
4. **Naive Bayes Classifier (Gaussian)**: A probabilistic classifier based on Bayes' theorem
5. **Random Forest Classifier**: An ensemble model combining multiple decision trees

## 6. Evaluation Metrics

Each model is evaluated using the following six metrics:

- **Accuracy**: Proportion of correctly classified instances
- **AUC Score**: Area Under the ROC Curve (weighted for multi-class)
- **Precision**: True positives among predicted positives (weighted average)
- **Recall**: True positives among actual positives (weighted average)
- **F1 Score**: Harmonic mean of precision and recall (weighted average)
- **Matthews Correlation Coefficient (MCC)**: Correlation coefficient between predicted and actual

## 7. Model Comparison Table

Results of model evaluation on the NPHA Doctor Visits held-out test dataset:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5524 | 0.6161 | 0.4672 | 0.5524 | 0.4487 | 0.1547 |
| Decision Tree | 0.3566 | 0.4630 | 0.3629 | 0.3566 | 0.3590 | -0.0585 |
| K-Nearest Neighbors | 0.4965 | 0.5399 | 0.4854 | 0.4965 | 0.4683 | 0.1083 |
| Naive Bayes | 0.2937 | 0.6217 | 0.4633 | 0.2937 | 0.2510 | 0.1427 |
| Random Forest | 0.5035 | 0.5600 | 0.4587 | 0.5035 | 0.4593 | 0.0942 |

## 8. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved the highest accuracy (0.5524) and MCC (0.1547). Although its F1 score (0.4487) was lower than KNN and Random Forest, the model provided the best overall accuracy, positive prediction agreement, interpretability, and computational efficiency. |
| **Decision Tree** | Performed relatively poorly, with the second-lowest accuracy (0.3566) and the lowest MCC (−0.0585). The negative MCC indicates weak agreement between the predicted and actual classes and may suggest overfitting or insufficient model tuning. |
| **K-Nearest Neighbors** | Achieved moderate performance with accuracy of 0.4965. Performance is sensitive to feature scaling and K value. Could be improved with tuning. |
| **Naive Bayes** | Lowest accuracy (0.2937) but surprisingly high AUC (0.6217). Feature independence assumption is violated. Strong ranking ability but poor hard classifications. |
| **Random Forest** | Achieved accuracy of 0.5035 with reasonable AUC (0.5600). Ensemble approach provides stability, though below Logistic Regression. |
| **Overall Winner for This Dataset** | **Logistic Regression** achieved the highest accuracy (0.5524) and MCC (0.1547). Although KNN achieved the highest F1 score (0.4683), Logistic Regression is selected as the overall preferred model because of its better overall accuracy, MCC, interpretability, and efficiency. |

## 9. Streamlit App Features

The Streamlit app includes the following required features:

✅ **CSV Dataset Upload Option**: Users can upload classification datasets (CSV format)
✅ **Target Column Selection**: Dropdown to select the target variable
✅ **Model Selection Dropdown**: Choose from 5 implemented models
✅ **Evaluation Metrics Display**: Table with all 6 metrics for all models
✅ **Confusion Matrix Display**: Visual heatmap and tabular format
✅ **Classification Report Display**: Detailed per-class metrics

**Additional Features**:
- Dataset preview (first 10 rows)
- Dataset statistics (rows, columns, target)
- Automatic preprocessing pipeline
- Error handling and validation
- Download metrics as CSV

## 10. Assignment Compliance Summary

- Dataset contains 714 instances, satisfying the minimum requirement of 500 instances.
- Dataset contains 14 input features, satisfying the minimum requirement of 12 features.
- Five required classification models are implemented:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbors Classifier
  4. Gaussian Naive Bayes Classifier
  5. Random Forest Classifier
- Six evaluation metrics are reported:
  Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Streamlit app includes CSV upload, target selection, model selection, confusion matrix, and classification report.
- GitHub repository and live Streamlit app links are provided.
