"""Train, evaluate, and save five classification pipelines for the NPHA dataset."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Number of Doctors Visited"
DATASET_PATH = Path("NPHA-doctor-visits.csv")
TEST_DATA_PATH = Path("test_data.csv")
MODEL_DIR = Path("models")
RESULTS_PATH = Path("model_comparison_results.csv")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path.resolve()}\n"
            "Place NPHA-doctor-visits.csv in the project root."
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing. "
            f"Available columns: {list(df.columns)}"
        )
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")


def replace_dataset_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the dataset's refused-response code with NaN.

    Review this list if a different version of the NPHA dataset is used.
    The target is deliberately excluded.
    """
    result = df.copy()
    columns_with_refused_code = [column for column in result.columns if column != TARGET_COLUMN]
    result[columns_with_refused_code] = result[columns_with_refused_code].replace(-1, np.nan)
    return result


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    if not transformers:
        raise ValueError("No usable predictor columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def multiclass_auc(y_true, probabilities, classes) -> float:
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


def safe_filename(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
        + ".joblib"
    )


def main() -> None:
    print("=" * 72)
    print("NPHA MODEL TRAINING")
    print("=" * 72)

    df = replace_dataset_missing_codes(load_dataset(DATASET_PATH))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if y.isna().any():
        raise ValueError("The target column contains missing values.")
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # This is the held-out file used by the Streamlit application.
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.to_csv(TEST_DATA_PATH, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    results = []
    model_paths = {}

    for model_name, estimator in get_models().items():
        print(f"\nTraining {model_name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        classes = pipeline.named_steps["classifier"].classes_

        metrics = {
            "ML Model Name": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": multiclass_auc(y_test, probabilities, classes),
            "Precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        results.append(metrics)

        model_path = MODEL_DIR / safe_filename(model_name)
        joblib.dump(pipeline, model_path)
        model_paths[model_name] = str(model_path)
        print(f"Saved: {model_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": X.columns.tolist(),
        "classes": sorted(y.unique().tolist()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model_paths": model_paths,
        "results_file": str(RESULTS_PATH),
        "test_data_file": str(TEST_DATA_PATH),
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")
    with open(MODEL_DIR / "index.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(results_df.round(4).to_string(index=False))
    print(f"\nHeld-out test data: {TEST_DATA_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Saved models: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
