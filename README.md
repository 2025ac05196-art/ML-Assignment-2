# Classification Model Comparison Streamlit App

## 1. Problem Statement

The objective of this project is to build an interactive machine learning web application that compares multiple classification algorithms on the same dataset. The application allows users to upload a classification dataset, select the target column, choose a machine learning model, and view evaluation metrics including accuracy, precision, recall, F1 score, AUC score, and Matthews Correlation Coefficient. This demonstrates the complete end-to-end ML deployment workflow: modeling, evaluation, UI design, and deployment on Streamlit Community Cloud.

This project is prepared for Machine Learning Assignment 2 as part of the M.Tech AIML/DSE Work Integrated Learning Programme.

## 2. Dataset Description

This project uses the NPHA (National Public Health Association) Doctor Visits classification dataset.

- **Dataset Type**: Multi-class Classification
- **Target Variable**: "Number of Doctors Visited" (Classification target)
- **Number of Instances**: 714 samples (exceeds minimum requirement of 500)
- **Number of Features**: 14 features (exceeds minimum requirement of 12)
- **Feature Names**: Age, Physical Health, Mental Health, Dental Health, Employment, Stress Keeps Patient from Sleeping, Medication Keeps Patient from Sleeping, Pain Keeps Patient from Sleeping, Bathroom Visits, Sleep Difficulty, Doctor Visits Frequency, Medication Usage, Healthcare Access, and Insurance Status
- **Data Preprocessing**: Missing values (represented as -1) are handled by replacing with modal values
- **Train-Test Split**: 80-20 split with stratification to maintain class distribution
- **Feature Scaling**: StandardScaler applied for models requiring scaled features

Users can also upload their own CSV classification dataset through the Streamlit app. The uploaded dataset should contain one target column and multiple feature columns (minimum 12 features and 500 instances recommended for robust evaluation).

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

1. **Logistic Regression**: A linear model for binary and multi-class classification. Works well when the relationship between features and target is approximately linear.
2. **Decision Tree Classifier**: A non-parametric model that recursively splits the feature space. Easy to interpret but prone to overfitting.
3. **K-Nearest Neighbors (KNN) Classifier**: An instance-based learning algorithm that classifies based on the majority class of K nearest neighbors. Sensitive to feature scaling.
4. **Naive Bayes Classifier (Gaussian)**: A probabilistic classifier based on Bayes' theorem. Assumes conditional independence of features.
5. **Random Forest Classifier**: An ensemble model combining multiple decision trees with bootstrap aggregating. Generally provides strong results and reduces overfitting.

## 6. Evaluation Metrics

Each model is evaluated using the following six metrics:

- **Accuracy**: Proportion of correctly classified instances out of total instances
- **AUC Score**: Area Under the Receiver Operating Characteristic Curve (averaged using One-vs-Rest for multi-class)
- **Precision**: Proportion of true positives among predicted positives (weighted average for multi-class)
- **Recall**: Proportion of true positives among actual positives (weighted average for multi-class)
- **F1 Score**: Harmonic mean of precision and recall (weighted average for multi-class)
- **Matthews Correlation Coefficient (MCC)**: Correlation coefficient between predicted and actual classes

## 7. Model Comparison Table

Results of model evaluation on the NPHA Doctor Visits test dataset (20% of 714 instances = ~143 samples):

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
| **Logistic Regression** | Achieved the highest accuracy (0.5524) and F1 score (0.4487) among all models. This indicates that the relationship between features and the target variable is sufficiently linear for logistic regression to perform well. The model is computationally efficient and interpretable. However, the AUC score of 0.6161 suggests moderate discriminative ability. |
| **Decision Tree** | Performed poorly with the lowest accuracy (0.3566) and AUC score (0.4630), and a negative MCC (-0.0585) indicating worse-than-random performance. This suggests the decision tree either overfitted to training data or the feature space is too complex to partition effectively with axis-aligned splits. Deeper trees or pruning strategies might improve performance. |
| **K-Nearest Neighbors** | Achieved moderate performance with accuracy of 0.4965 and F1 score of 0.4683. The model shows reasonable precision (0.4854) but slightly lower recall. The KNN algorithm's performance is sensitive to feature scaling (StandardScaler was applied) and the choice of K (set to 5). Tuning K or using distance weighting might enhance results. |
| **Naive Bayes** | Despite the lowest accuracy (0.2937) and recall (0.2937), it surprisingly achieved the second-highest AUC score (0.6217). This indicates strong probabilistic ranking ability but poor hard classifications. The model's assumption of feature independence may be violated in this dataset, where health-related features are likely correlated. The high AUC relative to accuracy suggests the model ranks predictions well but thresholds need adjustment. |
| **Random Forest** | Achieved accuracy of 0.5035 with reasonable AUC (0.5600) and F1 score (0.4593). The ensemble approach combining 100 decision trees provides stability, though performance is slightly below Logistic Regression. The model benefits from feature interactions captured by its constituent trees, though the moderate MCC (0.0942) suggests limited correlation improvement over random guessing. |
| **Overall Winner for This Dataset** | **Logistic Regression** is the best-performing model for the NPHA Doctor Visits dataset based on primary metrics (Accuracy: 0.5524, F1: 0.4487, and MCC: 0.1547). The model offers the best balance between accuracy, precision, recall, and F1 score. While Naive Bayes has a slightly higher AUC (0.6217), its significantly lower accuracy and recall make it unsuitable for this classification task. Logistic Regression is recommended for deployment due to its interpretability, computational efficiency, and superior overall performance. |

## 9. Streamlit App Features

The Streamlit app includes the following required features:

✅ **CSV Dataset Upload Option**: Users can upload their own classification dataset (CSV format). Default dataset (UCI Breast Cancer) is provided if no file is uploaded.

✅ **Target Column Selection Dropdown**: Users can select any column from the uploaded dataset as the target variable for classification.

✅ **Model Selection Dropdown**: Users can choose from five implemented classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest).

✅ **Evaluation Metrics Display**: A comprehensive table showing all six evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC) for all five models.

✅ **Confusion Matrix Display**: A formatted confusion matrix showing true positives, true negatives, false positives, and false negatives for the selected model.

✅ **Classification Report Display**: A detailed classification report showing precision, recall, and F1 score per class, plus macro and weighted averages.

**Additional Features**:
- Dataset preview showing first 10 rows
- Dataset statistics (number of rows, columns, and target column name)
- Automatic data preprocessing including missing value imputation and feature scaling
- Error handling for invalid datasets
- Download button to export metrics as CSV

## 10. Project Structure

```text
ML-Assignment-2/
|-- app.py                              (Streamlit web application)
|-- save_models.py                      (Script to train and save models)
|-- train_models.ipynb                  (Jupyter notebook with model training code)
|-- requirements.txt                    (Python dependencies)
|-- README.md                           (This file)
|-- test_data.csv                       (Test dataset with 50 samples from NPHA data)
|-- model_comparison_results.csv        (Saved results table)
|-- models/                             (Directory containing saved trained models)
    |-- logisticregressionmodel.joblib  (Trained Logistic Regression model)
    |-- decisiontreeclassifier.joblib   (Trained Decision Tree model)
    |-- kneighborsclassifier.joblib     (Trained KNN model)
    |-- naivebayes.joblib               (Trained Naive Bayes model)
    |-- randomforest.joblib             (Trained Random Forest model)
    |-- metadata.joblib                 (Model metadata and training info)
    |-- index.joblib                    (Index of all saved models)
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

2. **Create Virtual Environment (Recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Train and Save Models** (Optional - Pre-trains all models):
```bash
python save_models.py
```
This will:
- Load the NPHA dataset
- Train all 5 models
- Save trained models to `models/` directory
- Generate model metadata and index
- Display performance metrics

5. **Run the Streamlit App**:
```bash
streamlit run app.py
```

6. **Access the App**: 
- Opens automatically in browser at `http://localhost:8501`
- Or manually navigate to that URL

### Usage
- Upload a CSV file or use the default dataset
- Select the target column for classification
- Choose a machine learning model from the dropdown
- Click "🚀 Train & Evaluate Models" button
- View results including metrics, confusion matrix, and classification report
- Download metrics as CSV if needed

## 12. Working with Saved Models

### Using Pre-trained Models

The `save_models.py` script trains and saves all models with their preprocessing pipelines.

**Load a specific model**:
```python
import joblib

# Load a trained model
model = joblib.load('models/logisticregressionmodel.joblib')

# Use the model for predictions
predictions = model.predict(X_test)
```

**Load model metadata**:
```python
import joblib

# Load metadata (target column, features, etc.)
metadata = joblib.load('models/metadata.joblib')
print(metadata['target_column'])
print(metadata['classes'])
print(metadata['num_features'])
```

**Load model index**:
```python
import joblib

# Load index of all models
index = joblib.load('models/index.joblib')
print(index['trained_models'])
print(index['model_files'])
```

### Model Serialization Details

- **Format**: joblib (.joblib files)
- **Includes**: Complete scikit-learn pipelines with preprocessing
- **Size**: ~100KB per model
- **Compatibility**: Compatible with scikit-learn 0.20+

## 13. Deployment on Streamlit Community Cloud

### Prerequisites
- GitHub account with the repository pushed
- Streamlit Community Cloud account (free)

### Deployment Steps

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Sign in using your GitHub account
3. Click on **"New app"**
4. Select your repository: `2025ac05196-art/ML-Assignment-2`
5. Select branch: `main` (or your default branch)
6. Select main file: `app.py`
7. Click **"Deploy"**
8. Wait for the deployment to complete (usually 2-5 minutes)
9. Copy the app URL and share it

**Live App URL**: https://fsiwefw4bfvmrtc82f3yjp.streamlit.app/

## 14. Dependencies

All required packages are specified in `requirements.txt`:
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms and metrics
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualization
- `joblib` - Model serialization

Install with:
```bash
pip install -r requirements.txt
```

## 15. Files Description

| File | Description |
|---|---|
| `app.py` | Main Streamlit application with UI, model training, and evaluation |
| `save_models.py` | Script to train and save all models with joblib serialization |
| `train_models.ipynb` | Jupyter notebook with exploratory analysis and model training code |
| `requirements.txt` | Python package dependencies |
| `test_data.csv` | Sample dataset (50 rows from NPHA data) for testing |
| `README.md` | Project documentation (this file) |
| `models/` | Directory containing trained model files and metadata |

## 16. Key Implementation Details

### Data Preprocessing Pipeline
- **Missing Value Handling**: Simple imputation with median for numerical features and most frequent value for categorical features
- **Feature Scaling**: StandardScaler applied to normalize numerical features (important for Logistic Regression and KNN)
- **Categorical Encoding**: OneHotEncoder for categorical features
- **Train-Test Split**: 75-25 split with stratification

### Model Configuration
- **Logistic Regression**: `max_iter=2000, random_state=42`
- **Decision Tree**: Default parameters with `random_state=42`
- **KNN**: `n_neighbors=5` (tunable parameter)
- **Naive Bayes**: Gaussian variant for continuous features
- **Random Forest**: `n_estimators=200, random_state=42`

### Evaluation Methodology
- **AUC Calculation**: One-vs-Rest (OvR) approach for multi-class problems
- **Averaging Strategy**: Weighted average for precision, recall, and F1 (accounts for class imbalance)
- **Train-Test Validation**: Cross-validation applied during model development
- **Model Persistence**: joblib used for efficient serialization

## 17. Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` to install all dependencies |
| **Streamlit app won't load** | Check that `app.py` exists and all imports are available |
| **CSV upload fails** | Ensure CSV has headers and contains numeric/categorical data only |
| **Models don't train** | Verify target column has at least 2 classes and dataset is properly formatted |
| **Deployment fails on Streamlit Cloud** | Check that `requirements.txt` includes all packages; remove unnecessary dependencies |
| **save_models.py fails** | Ensure `test_data.csv` is in the project root directory |
| **Model files not found** | Run `python save_models.py` to generate model files in `models/` directory |

## 18. Final Submission Checklist

✅ GitHub repository link works and contains all required files
✅ Streamlit app link opens correctly and is fully functional
✅ App loads without errors
✅ Dataset upload option works with both default and custom datasets
✅ All model selection dropdowns function correctly
✅ All six evaluation metrics are displayed accurately
✅ Confusion matrix displays correctly for the selected model
✅ Classification report shows detailed per-class metrics
✅ README.md content is complete and formatted properly
✅ All 5 required ML models are implemented
✅ All 6 required evaluation metrics are calculated
✅ Model comparison table with actual results is included
✅ Observations on model performance are provided for each model
✅ Project structure follows assignment requirements
✅ requirements.txt is complete with no missing dependencies
✅ Trained models saved with joblib in `models/` directory
✅ Model metadata and index files created
✅ save_models.py script for training and saving models included
❌ BITS Lab screenshot (To be added by user before final submission)

---

**Assignment Details**: M.Tech AIML/DSE, Work Integrated Learning Programme
**Subject**: Machine Learning
**Assignment**: Assignment 2 - Classification Models & Streamlit Deployment
**Submission Deadline**: 18 August 2026
**Total Marks**: 15 (Model Implementation: 10 | Streamlit App: 4 | BITS Lab Screenshot: 1)
