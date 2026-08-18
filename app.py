"""Streamlit application for evaluating saved NPHA classification models."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

MODEL_DIR = Path("models")
METADATA_PATH = MODEL_DIR / "metadata.joblib"
DEFAULT_TEST_PATH = Path("test_data.csv")

st.set_page_config(
    page_title="NPHA Classification Model Comparison",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_project_artifacts():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "models/metadata.joblib was not found. Run `python save_models.py` first."
        )
    metadata = joblib.load(METADATA_PATH)
    models = {}
    for name, path in metadata["model_paths"].items():
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        models[name] = joblib.load(model_path)
    return metadata, models


@st.cache_data
def load_default_test_data():
    if DEFAULT_TEST_PATH.exists():
        return pd.read_csv(DEFAULT_TEST_PATH)
    return None


def clean_input(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    result = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    predictor_columns = [c for c in result.columns if c != target_column]
    result[predictor_columns] = result[predictor_columns].replace(-1, np.nan)
    return result


def calculate_auc(y_true, probabilities, classes) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1], labels=classes))
        y_true_bin = label_binarize(y_true, classes=classes)
        return float(
            roc_auc_score(
                y_true_bin,
                probabilities,
                average="weighted",
                multi_class="ovr",
            )
        )
    except ValueError:
        return float("nan")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["classifier"].classes_
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": calculate_auc(y_test, probabilities, classes),
        "Precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return metrics, y_pred, classes


st.title("🤖 NPHA Classification Model Comparison")
st.write(
    "Evaluate one or all five saved classification models on the held-out "
    "NPHA test dataset. The uploaded CSV must contain the target column to "
    "calculate metrics, a confusion matrix, and a classification report."
)

try:
    metadata, saved_models = load_project_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("Run `python save_models.py`, commit the generated models folder, and redeploy.")
    st.stop()

target_column = metadata["target_column"]
expected_features = metadata["feature_columns"]

with st.sidebar:
    st.header("⚙️ Controls")
    source = st.radio("Data source", ["Use held-out test data", "Upload CSV"])
    selected_model_name = st.selectbox("Select model", list(saved_models.keys()))
    compare_all = st.checkbox("Compare all models", value=True)
    run_evaluation = st.button("🚀 Evaluate", type="primary", use_container_width=True)

if source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    dataset = pd.read_csv(uploaded_file) if uploaded_file is not None else None
else:
    dataset = load_default_test_data()

if dataset is None:
    st.warning("Select the held-out data or upload a CSV file to continue.")
    st.stop()

try:
    dataset = clean_input(dataset, target_column)
except Exception as exc:
    st.error(f"Could not clean the dataset: {exc}")
    st.stop()

st.subheader("📌 Test Dataset Preview")
st.dataframe(dataset.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(dataset))
col2.metric("Columns", dataset.shape[1])
col3.metric("Target", target_column if target_column in dataset.columns else "Missing")

missing_features = [c for c in expected_features if c not in dataset.columns]
extra_features = [c for c in dataset.columns if c not in expected_features + [target_column]]
if missing_features:
    st.error(f"Missing required feature columns: {missing_features}")
    st.stop()
if extra_features:
    st.info(f"Extra columns will be ignored: {extra_features}")
if target_column not in dataset.columns:
    st.error(
        f"The uploaded test file must include the target column '{target_column}' "
        "for evaluation."
    )
    st.stop()

X_test = dataset[expected_features]
y_test = dataset[target_column]
if y_test.isna().any():
    st.error("The target column contains missing values.")
    st.stop()

if run_evaluation:
    model_names = list(saved_models.keys()) if compare_all else [selected_model_name]
    rows = []
    evaluation_details = {}

    with st.spinner("Evaluating saved models..."):
        for name in model_names:
            try:
                metrics, y_pred, classes = evaluate_model(
                    saved_models[name], X_test, y_test
                )
                rows.append({"ML Model Name": name, **metrics})
                evaluation_details[name] = (y_pred, classes)
            except Exception as exc:
                st.warning(f"Could not evaluate {name}: {exc}")

    if not rows:
        st.error("No model could be evaluated successfully.")
        st.stop()

    metrics_df = pd.DataFrame(rows)
    st.subheader("📊 Model Comparison Metrics")
    st.dataframe(
        metrics_df.style.format(
            {column: "{:.4f}" for column in metrics_df.columns if column != "ML Model Name"}
        ),
        use_container_width=True,
    )
    st.download_button(
        "📥 Download Metrics CSV",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_metrics.csv",
        mime="text/csv",
    )

    detail_name = (
        selected_model_name
        if selected_model_name in evaluation_details
        else next(iter(evaluation_details))
    )
    y_pred, classes = evaluation_details[detail_name]
    labels = list(classes)

    st.subheader(f"🔍 Detailed Results: {detail_name}")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[str(value) for value in labels],
        zero_division=0,
        output_dict=True,
    )

    left, right = st.columns(2)
    with left:
        st.write("**Confusion Matrix**")
        figure, axis = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(f"Confusion Matrix - {detail_name}")
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with right:
        st.write("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    selected_row = metrics_df.loc[metrics_df["ML Model Name"] == detail_name].iloc[0]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Accuracy", f"{selected_row['Accuracy']:.4f}")
    metric_columns[1].metric("F1 Score", f"{selected_row['F1 Score']:.4f}")
    metric_columns[2].metric(
        "AUC", "N/A" if pd.isna(selected_row["AUC"]) else f"{selected_row['AUC']:.4f}"
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Precision", f"{selected_row['Precision']:.4f}")
    metric_columns[1].metric("Recall", f"{selected_row['Recall']:.4f}")
    metric_columns[2].metric("MCC", f"{selected_row['MCC']:.4f}")
    st.success("Evaluation completed successfully.")
else:
    st.info("Choose the controls in the sidebar and click Evaluate.")

st.divider()
st.markdown(
    """
### About this project
- Machine Learning Assignment 2 — WILP M.Tech (AIML/DSE)
- Five classification models evaluated on the same held-out test set
- Six metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- [GitHub Repository](https://github.com/2025ac05196-art/ML-Assignment-2)
- [Live Streamlit Application](https://fsiwefw4bfvmrtc82f3yjp.streamlit.app/)
"""
)
