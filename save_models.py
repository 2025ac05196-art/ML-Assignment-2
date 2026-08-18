import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    label_binarize
)

# ==========================================
# CONFIGURATION
# ==========================================

RANDOM_STATE = 42
MODEL_DIR = "models"
DATASET_PATH = "test_data.csv"

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"✅ Created '{MODEL_DIR}' directory")

# ==========================================
# LOAD AND PREPARE DATASET
# ==========================================

print("\n" + "="*70)
print("LOADING NPHA DOCTOR VISITS DATASET")
print("="*70)

if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: {DATASET_PATH} not found!")
    print("\nPlease ensure test_data.csv is in the current directory.")
    exit(1)

df = pd.read_csv(DATASET_PATH)

print(f"✅ Dataset loaded successfully")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# ==========================================
# DATA PREPROCESSING
# ==========================================

print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Handle missing values
df.replace(-1, np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

print(f"✅ Missing values handled")

# Define target column
target_column = "Number of Doctors Visited"

if target_column not in df.columns:
    print(f"❌ Error: Target column '{target_column}' not found!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"✅ Classes: {sorted(y.unique())}")

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"✅ Training set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")

# ==========================================
# BUILD PREPROCESSING PIPELINE
# ==========================================

print("\n" + "="*70)
print("BUILDING PREPROCESSING PIPELINE")
print("="*70)

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessors = []
if numeric_features:
    preprocessors.append(("num", numeric_transformer, numeric_features))
if categorical_features:
    preprocessors.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=preprocessors)

print(f"✅ Numeric features: {len(numeric_features)}")
print(f"✅ Categorical features: {len(categorical_features)}")

# ==========================================
# FEATURE SCALING (for scaled models)
# ==========================================

print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled successfully")

# ==========================================
# MODEL DEFINITIONS
# ==========================================

def get_models():
    """Return dictionary of all ML models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


class DenseTransformer:
    """Convert sparse matrix output to dense array for GaussianNB."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


# ==========================================
# MODEL TRAINING AND SAVING
# ==========================================

print("\n" + "="*70)
print("TRAINING AND SAVING MODELS")
print("="*70)

models = get_models()
results = []
model_paths = {}

for model_name, estimator in models.items():
    print(f"\n🔄 Training {model_name}...")

    try:
        if model_name == "Naive Bayes":
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("to_dense", DenseTransformer()),
                    ("classifier", estimator),
                ]
            )
        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", estimator),
                ]
            )

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate AUC
        try:
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)
                classes = pipeline.classes_
                if len(classes) == 2:
                    auc = roc_auc_score(y_test, y_score[:, 1])
                else:
                    y_test_bin = label_binarize(y_test, classes=list(range(len(classes))))
                    auc = roc_auc_score(y_test_bin, y_score, average="weighted", multi_class="ovr")
            else:
                auc = np.nan
        except:
            auc = np.nan

        # Store results
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
        })

        # Save the model
        model_filename = f"{model_name.lower().replace(' ', '_').replace('-', '')}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(pipeline, model_path)
        model_paths[model_name] = model_path

        print(f"   ✅ Trained successfully")
        print(f"   📊 Accuracy: {accuracy:.4f}")
        print(f"   📊 F1 Score: {f1:.4f}")
        print(f"   📊 AUC: {auc:.4f}")
        print(f"   💾 Model saved to: {model_path}")

    except Exception as e:
        print(f"   ❌ Error training {model_name}: {str(e)}")
        continue

# ==========================================
# SAVE RESULTS
# ==========================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_path = "model_comparison_results.csv"
results_df.to_csv(results_path, index=False)

print(f"✅ Results saved to: {results_path}")

# ==========================================
# MODEL COMPARISON TABLE
# ==========================================

print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print(results_df.to_string(index=False))

# ==========================================
# SAVE METADATA
# ==========================================

print("\n" + "="*70)
print("SAVING MODEL METADATA")
print("="*70)

metadata = {
    "target_column": target_column,
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "num_features": X_train.shape[1],
    "num_classes": len(y.unique()),
    "classes": sorted(y.unique()),
    "random_state": RANDOM_STATE,
    "model_paths": model_paths,
    "preprocessing": {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
}

metadata_path = os.path.join(MODEL_DIR, "metadata.joblib")
joblib.dump(metadata, metadata_path)

print(f"✅ Metadata saved to: {metadata_path}")

# ==========================================
# CREATE MODEL INDEX
# ==========================================

print("\n" + "="*70)
print("CREATING MODEL INDEX")
print("="*70)

model_index = {
    "model_files": model_paths,
    "metadata_file": metadata_path,
    "results_file": results_path,
    "target_column": target_column,
    "num_models": len(model_paths),
    "trained_models": list(model_paths.keys()),
}

index_path = os.path.join(MODEL_DIR, "index.joblib")
joblib.dump(model_index, index_path)

print(f"✅ Model index created: {index_path}")
print(f"\n📁 Models saved:")
for model_name, path in model_paths.items():
    print(f"   - {model_name}: {path}")

# ==========================================
# COMPLETION SUMMARY
# ==========================================

print("\n" + "="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
print(f"\n📊 Summary:")
print(f"   - Models trained: {len(model_paths)}")
print(f"   - Models saved to: {MODEL_DIR}/")
print(f"   - Results saved to: {results_path}")
print(f"   - Metadata saved to: {metadata_path}")
print(f"   - Model index saved to: {index_path}")
print(f"\n🚀 Next steps:")
print(f"   1. Models are ready for deployment")
print(f"   2. Use 'models/index.joblib' to load model information")
print(f"   3. Use 'models/<model>.joblib' to load individual models")
print(f"   4. Run 'streamlit run app.py' to start the Streamlit app")
