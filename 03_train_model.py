import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance

from utils import TARGET_COL


DATA_PATH = "data/processed/credit_clean.csv"
MODEL_PATH = "models/best_model.pkl"
METADATA_PATH = "models/model_metadata.json"

MODEL_COMPARISON_PATH = "reports/model_comparison.csv"
THRESHOLD_METRICS_PATH = "reports/threshold_metrics.csv"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"
TEST_PREDICTIONS_PATH = "reports/test_predictions.csv"


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba)
    }

    return metrics, y_pred, y_proba


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Cleaned data not found. Run python 01_clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in cleaned dataset.")

    X = df.drop(columns=["id", TARGET_COL])
    y = df[TARGET_COL]

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True) * 100)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        )
    }

    results = []
    trained_pipelines = {}
    prediction_store = {}

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print("Training:", model_name)
        print("=" * 80)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(
            pipeline,
            X_test,
            y_test,
            threshold=0.5
        )

        result = {"model": model_name}
        result.update(metrics)
        results.append(result)

        trained_pipelines[model_name] = pipeline
        prediction_store[model_name] = y_proba

        print("Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by="pr_auc", ascending=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    print("\nModel comparison saved to:", MODEL_COMPARISON_PATH)
    print(comparison_df)

    best_model_name = comparison_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_y_proba = prediction_store[best_model_name]

    print("\nBest model selected:", best_model_name)

    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "best_model": best_model_name,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COL,
        "selection_metric": "pr_auc",
        "default_threshold": 0.35
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Best model saved to:", MODEL_PATH)
    print("Metadata saved to:", METADATA_PATH)

    test_predictions = pd.DataFrame({
        "y_true": y_test.values,
        "y_proba": best_y_proba
    })

    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    threshold_rows = []

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred_threshold = (best_y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

        threshold_rows.append({
            "threshold": round(threshold, 2),
            "precision": precision_score(y_test, y_pred_threshold, zero_division=0),
            "recall": recall_score(y_test, y_pred_threshold, zero_division=0),
            "f1": f1_score(y_test, y_pred_threshold, zero_division=0),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "customers_flagged": int(y_pred_threshold.sum())
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(THRESHOLD_METRICS_PATH, index=False)

    print("Threshold metrics saved to:", THRESHOLD_METRICS_PATH)

    print("\nCalculating permutation feature importance...")

    sample_size = min(3000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    importance = permutation_importance(
        best_pipeline,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print("Feature importance saved to:", FEATURE_IMPORTANCE_PATH)
    print(importance_df.head(15))


if __name__ == "__main__":
    main()