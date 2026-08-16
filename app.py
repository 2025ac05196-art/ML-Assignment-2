import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
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
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Classification Model Comparison App",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Classification Model Comparison App")
st.write(
    "Upload a classification dataset, choose the target column, select a machine learning model, "
    "and view evaluation metrics, confusion matrix, and classification report."
)

RANDOM_STATE = 42


@st.cache_data
def load_default_dataset():
    """Load the UCI Wisconsin Diagnostic Breast Cancer dataset from scikit-learn."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "diagnosis"}, inplace=True)
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    return df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def calculate_auc(model, X_test, y_test, labels):
    """Calculate AUC for binary or multi-class classification."""
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            return np.nan

        if len(labels) == 2:
            return roc_auc_score(y_test, y_score[:, 1])

        y_test_bin = label_binarize(y_test, classes=list(range(len(labels))))
        return roc_auc_score(y_test_bin, y_score, average="weighted", multi_class="ovr")
    except Exception:
        return np.nan


def fit_and_evaluate(df, target_col, selected_model_name):
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    class_names = label_encoder.classes_

    stratify_value = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify_value,
    )

    models = get_models()
    results = []
    fitted_models = {}

    preprocessor = build_preprocessor(X)

    for model_name, estimator in models.items():
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

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        auc_value = calculate_auc(pipeline, X_test, y_test, class_names)

        results.append(
            {
                "ML Model Name": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": auc_value,
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "MCC": matthews_corrcoef(y_test, y_pred),
            }
        )
        fitted_models[model_name] = (pipeline, X_test, y_test, y_pred, class_names)

    selected_model = fitted_models[selected_model_name]
    metrics_df = pd.DataFrame(results)
    return metrics_df, selected_model


class DenseTransformer:
    """Convert sparse matrix output to dense array for GaussianNB."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


with st.sidebar:
    st.header("⚙️ App Controls")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded_file is None:
        st.info("Using default UCI Breast Cancer dataset. You can upload your own CSV file.")
        dataset = load_default_dataset()
    else:
        dataset = pd.read_csv(uploaded_file)

    st.subheader("Dataset Settings")
    target_column = st.selectbox("Select target column", dataset.columns, index=len(dataset.columns) - 1)

    model_name = st.selectbox(
        "Select model",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
        ],
    )

st.subheader("📌 Dataset Preview")
st.dataframe(dataset.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", dataset.shape[0])
col2.metric("Columns", dataset.shape[1])
col3.metric("Target", target_column)

if dataset[target_column].nunique() < 2:
    st.error("The selected target column must contain at least two classes.")
else:
    try:
        metrics_table, selected = fit_and_evaluate(dataset, target_column, model_name)
        model, X_test, y_test, y_pred, class_names = selected

        st.subheader("📊 Model Comparison Metrics")
        formatted_metrics = metrics_table.copy()
        numeric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        for col in numeric_cols:
            formatted_metrics[col] = formatted_metrics[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.4f}")
        st.dataframe(formatted_metrics, use_container_width=True)

        st.subheader(f"🔍 Detailed Results: {model_name}")

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        left, right = st.columns(2)
        with left:
            st.write("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)

        with right:
            st.write("**Classification Report**")
            report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.success("Evaluation completed successfully.")

    except Exception as exc:
        st.error("Unable to train/evaluate models. Please check that your CSV is suitable for classification.")
        st.exception(exc)
