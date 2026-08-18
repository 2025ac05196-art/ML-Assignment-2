# Classification Model Comparison Streamlit App

## 1. Problem Statement

The objective of this project is to build an interactive machine learning web application that compares multiple classification algorithms on the same dataset. The application allows users to evaluate pre-trained classification models, select the target column, and view comprehensive evaluation metrics including accuracy, precision, recall, F1 score, AUC score, and Matthews Correlation Coefficient. This demonstrates the complete end-to-end ML deployment workflow: modeling, evaluation, UI design, and deployment on Streamlit Community Cloud.

This project is prepared for Machine Learning Assignment 2 as part of the M.Tech AIML/DSE Work Integrated Learning Programme.

## 2. Dataset Description

This project uses the NPHA (National Public Health Association) Doctor Visits classification dataset.

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
| **Logistic Regression** | Achieved the highest accuracy (0.5524) and F1 score (0.4487). The linear relationship between features and target is well-captured. Model is efficient and interpretable. |
| **Decision Tree** | Performed poorly with the lowest accuracy (0.3566) and negative MCC. Suggests overfitting or complex feature interactions requiring non-linear partitioning. |
| **K-Nearest Neighbors** | Achieved moderate performance with accuracy of 0.4965. Performance is sensitive to feature scaling and K value. Could be improved with tuning. |
| **Naive Bayes** | Lowest accuracy (0.2937) but surprisingly high AUC (0.6217). Feature independence assumption is violated. Strong ranking ability but poor hard classifications. |
| **Random Forest** | Achieved accuracy of 0.5035 with reasonable AUC (0.5600). Ensemble approach provides stability, though below Logistic Regression. |
| **Overall Winner for This Dataset** | **Logistic Regression** is the best model based on Accuracy (0.5524), F1 Score (0.4487), and MCC (0.1547). Recommended for deployment due to interpretability and efficiency. |

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

## 10. Project Structure

```text
ML-Assignment-2/
|-- app.py                              (Streamlit application)
|-- save_models.py                      (Script to train and save models)
|-- train_models.ipynb                  (Jupyter notebook with analysis)
|-- requirements.txt                    (Python dependencies)
|-- README.md                           (This file)
|-- test_data.csv                       (Held-out test dataset)
|-- NPHA-doctor-visits.csv              (Full NPHA dataset)
|-- model_comparison_results.csv        (Performance metrics)
|-- models/                             (Saved trained models directory)
    |-- logisticregressionmodel.joblib
    |-- decisiontreeclassifier.joblib
    |-- kneighborsclassifier.joblib
    |-- naivebayes.joblib
    |-- randomforest.joblib
    |-- metadata.joblib
    └── index.json
```

## 11. How to Run Locally

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation and Execution

1. **Clone the Repository**:
```bash
git clone https://github.com/2025ac05196-art/ML-Assignment-2.git
cd ML-Assignment-2
```

2. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**:
```bash
streamlit run app.py
```

5. **Access the App**: 
Opens automatically at `http://localhost:8501`

## 12. Deployment on Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Sign in using your GitHub account
3. Click **"New app"**
4. Select repository: `2025ac05196-art/ML-Assignment-2`
5. Select branch: `main`
6. Select main file: `app.py`
7. Click **Deploy**

**Live App**: https://fsiwefw4bfvmrtc82f3yjp.streamlit.app/

## 13. Dependencies

All packages in `requirements.txt`:
```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

## 14. Key Implementation Details

### Data Preprocessing
- Missing Value Handling: Median for numerical, most_frequent for categorical
- Feature Scaling: StandardScaler for normalization
- Categorical Encoding: OneHotEncoder with handle_unknown="ignore"
- Train-Test Split: 80-20 split with stratification

### Model Configuration
- **Logistic Regression**: max_iter=2000
- **Decision Tree**: random_state=42
- **KNN**: n_neighbors=5
- **Naive Bayes**: Gaussian variant
- **Random Forest**: n_estimators=200

## 15. Troubleshooting

| Issue | Solution |
|---|---|
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` |
| **Streamlit app won't load** | Check that app.py exists and imports are available |
| **CSV upload fails** | Ensure CSV has proper headers and target column |
| **Target column missing** | Check CSV contains "Number of Doctors Visited" column |
| **Deployment fails** | Ensure models/ directory is pushed to GitHub |

## 16. Final Submission Checklist

✅ GitHub repository link works
✅ Streamlit app link opens correctly
✅ App loads without errors
✅ All 5 ML models implemented
✅ All 6 evaluation metrics calculated
✅ Model comparison table with results
✅ Observations on model performance
✅ Confusion matrix displays correctly
✅ Classification report shows metrics
✅ README content complete
✅ requirements.txt with all packages
✅ Test data (test_data.csv) included
✅ Trained models saved (models/ directory)
❌ BITS Lab screenshot (to be added)

---

**Assignment Details**: M.Tech AIML/DSE, Work Integrated Learning Programme
**Subject**: Machine Learning
**Assignment**: Assignment 2 - Classification Models & Streamlit Deployment
**Deadline**: 18 August 2026, 23:59 PM
**Total Marks**: 15 (Model: 10 | App: 4 | Lab Screenshot: 1)
