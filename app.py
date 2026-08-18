
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
def load_npha_dataset():
    """Load the NPHA Doctor Visits dataset."""
    try:
        df = pd.read_csv("test_data.csv")
        return df
    except:
        return None


def build_preprocessor(X):
    """Build preprocessing pipeline for numerical and categorical features."""
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessors = []
    if numeric_features:
        preprocessors.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        preprocessors.append(("cat", categorical_transformer, categorical_features))

    if not preprocessors:
        return None

    return ColumnTransformer(transformers=preprocessors)


def get_models():
    """Return dictionary of all ML models."""
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


class DenseTransformer:
    """Convert sparse matrix output to dense array for GaussianNB."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


def fit_and_evaluate(df, target_col, selected_model_name):
    """Train all models and return metrics comparison and selected model results."""
    # Data cleaning
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Handle missing values in target
    y_raw = y_raw.fillna(y_raw.mode()[0] if not y_raw.mode().empty else y_raw.iloc[0])

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    class_names = label_encoder.classes_

    # Determine stratification
    stratify_value = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify_value,
    )

    # Build preprocessor
    preprocessor = build_preprocessor(X_train)
    if preprocessor is None:
        raise ValueError("Could not build preprocessor for the dataset")

    # Train all models
    models = get_models()
    results = []
    fitted_models = {}

    for model_name, estimator in models.items():
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

            # Train model
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            auc_value = calculate_auc(pipeline, X_test, y_test, class_names)

            # Calculate metrics
            results.append(
                {
                    "ML Model Name": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC": auc_value,
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "MCC": matthews_corrcoef(y_test, y_pred),
                }
            )
            fitted_models[model_name] = (pipeline, X_test, y_test, y_pred, class_names)
        except Exception as e:
            st.warning(f"Error training {model_name}: {str(e)}")
            continue

    if selected_model_name not in fitted_models:
        selected_model_name = list(fitted_models.keys())[0] if fitted_models else None
        
    if selected_model_name is None or selected_model_name not in fitted_models:
        raise ValueError("Could not train any model successfully")

    selected_model = fitted_models[selected_model_name]
    metrics_df = pd.DataFrame(results)
    return metrics_df, selected_model, selected_model_name


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ App Controls")
    
    # Data source selection
    data_source = st.radio("Select Data Source", ["Upload CSV", "Use Test Data"])
    
    dataset = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                dataset = pd.read_csv(uploaded_file)
                st.success("✅ Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                dataset = None
        else:
            st.info("Please upload a CSV file to continue.")
    else:
        # Try to load test data
        npha_df = load_npha_dataset()
        if npha_df is not None:
            dataset = npha_df
            st.success("✅ Test dataset (NPHA) loaded successfully!")
        else:
            st.warning("test_data.csv not found. Please upload a CSV file.")

    # If dataset is loaded, show settings
    target_column = None
    model_name = None
    train_button = False
    
    if dataset is not None and not dataset.empty:
        st.subheader("Dataset Settings")
        
        # Target column selection
        target_column = st.selectbox(
            "Select target column",
            dataset.columns,
            index=min(len(dataset.columns) - 1, 0)
        )

        # Model selection
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
        
        # Train button
        train_button = st.button("🚀 Train & Evaluate Models", type="primary")
    else:
        st.info("👈 Upload a CSV or use test data to get started")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if dataset is not None and not dataset.empty:
    # Dataset preview
    st.subheader("📌 Dataset Preview")
    st.dataframe(dataset.head(10), use_container_width=True)

    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Rows", dataset.shape[0])
    col2.metric("🔢 Columns", dataset.shape[1])
    col3.metric("🎯 Target", target_column if target_column else "N/A")

    # Show missing values
    missing_info = dataset.isnull().sum()
    if missing_info.any():
        with st.expander("⚠️ Missing Values Info"):
            st.dataframe(missing_info[missing_info > 0])

    # Validate target column
    if target_column:
        unique_classes = dataset[target_column].nunique()
        if unique_classes < 2:
            st.error(f"❌ The selected target column has only {unique_classes} class(es). Classification requires at least 2 classes.")
        elif train_button:
            # Train and evaluate
            try:
                with st.spinner("🔄 Training models... This may take a moment."):
                    metrics_table, selected, actual_model_name = fit_and_evaluate(dataset, target_column, model_name)
                    model, X_test, y_test, y_pred, class_names = selected

                # Model comparison metrics
                st.subheader("📊 Model Comparison Metrics")
                formatted_metrics = metrics_table.copy()
                numeric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
                for col in numeric_cols:
                    if col in formatted_metrics.columns:
                        formatted_metrics[col] = formatted_metrics[col].map(
                            lambda x: "N/A" if pd.isna(x) else f"{x:.4f}"
                        )
                st.dataframe(formatted_metrics, use_container_width=True)

                # Download results
                csv = metrics_table.to_csv(index=False)
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=csv,
                    file_name="model_comparison_metrics.csv",
                    mime="text/csv"
                )

                # Detailed results for selected model
                st.subheader(f"🔍 Detailed Results: {actual_model_name}")

                # Get predictions and confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Ensure confusion matrix dimensions match class names
                num_classes = len(class_names)
                if cm.shape[0] != num_classes or cm.shape[1] != num_classes:
                    st.warning(f"Note: Expected {num_classes}x{num_classes} confusion matrix, got {cm.shape}")
                    cm_padded = np.zeros((num_classes, num_classes), dtype=int)
                    min_dim = min(cm.shape[0], cm.shape[1], num_classes)
                    cm_padded[:min_dim, :min_dim] = cm[:min_dim, :min_dim]
                    cm = cm_padded
                
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Confusion Matrix**")
                    st.dataframe(cm_df, use_container_width=True)
                    
                    # Visualize confusion matrix
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                               xticklabels=class_names, yticklabels=class_names)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix - {actual_model_name}")
                    st.pyplot(fig, use_container_width=True)

                with col2:
                    st.write("**Classification Report**")
                    try:
                        report = classification_report(
                            y_test, y_pred,
                            target_names=class_names,
                            zero_division=0,
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate classification report: {e}")

                # Individual model metrics
                st.subheader(f"📈 {actual_model_name} Performance Metrics")
                selected_metrics = metrics_table[metrics_table["ML Model Name"] == actual_model_name].iloc[0]
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
                metric_col2.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
                metric_col3.metric("AUC Score", f"{selected_metrics['AUC']:.4f}")

                metric_col4, metric_col5, metric_col6 = st.columns(3)
                metric_col4.metric("Precision", f"{selected_metrics['Precision']:.4f}")
                metric_col5.metric("Recall", f"{selected_metrics['Recall']:.4f}")
                metric_col6.metric("MCC", f"{selected_metrics['MCC']:.4f}")

                st.success("✅ Evaluation completed successfully!")

            except Exception as exc:
                st.error("❌ Unable to train/evaluate models.")
                st.error(f"Error Details: {str(exc)}")
                with st.expander("📋 Full Error Traceback"):
                    st.exception(exc)
        else:
            st.info(f"👆 Select a model and click '🚀 Train & Evaluate Models' to start (Classes: {unique_classes})")
else:
    st.warning("📁 Please upload a CSV file or select test data from the sidebar to begin.")

# Footer
st.divider()
st.markdown("""
**📚 About This App**
- Machine Learning Assignment 2 - WILP M.Tech (AIML/DSE)
- Compares 5 classification models on the same dataset
- Provides comprehensive evaluation metrics and visualizations
- Deploy on Streamlit Community Cloud

**🔗 Links:**
- [GitHub Repository](https://github.com/2025ac05196-art/ML-Assignment-2)
- [Test Dataset](https://github.com/2025ac05196-art/ML-Assignment-2/blob/main/test_data.csv)
""")
