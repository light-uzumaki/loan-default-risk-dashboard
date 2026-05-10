import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from utils import clean_credit_data, TARGET_COL


DATA_PATH = "data/processed/credit_clean.csv"
MODEL_PATH = "models/best_model.pkl"
METADATA_PATH = "models/model_metadata.json"
MODEL_COMPARISON_PATH = "reports/model_comparison.csv"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"
TEST_PREDICTIONS_PATH = "reports/test_predictions.csv"


st.set_page_config(
    page_title="Loan Default Risk Dashboard",
    page_icon="💳",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_reports():
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    test_preds = pd.read_csv(TEST_PREDICTIONS_PATH)
    return comparison, importance, test_preds


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def calculate_threshold_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "flagged": int(y_pred.sum())
    }


df = load_data()
model = load_model()
metadata = load_metadata()
comparison_df, importance_df, test_preds = load_reports()


st.title("💳 Loan Default Risk Prediction & Analysis Dashboard")

st.write(
    "This dashboard analyzes customer credit behavior and predicts the probability "
    "of next-month default using machine learning models built with scikit-learn."
)

st.warning(
    "Educational project only: this is not a real banking approval system."
)


page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Overview",
        "Risk Analysis",
        "Model Performance",
        "Threshold Simulator",
        "Customer Risk Predictor"
    ]
)


if page == "Executive Overview":
    st.header("Executive Overview")

    total_customers = len(df)
    default_rate = df[TARGET_COL].mean() * 100
    avg_credit_limit = df["limit_bal"].mean()
    avg_age = df["age"].mean()
    delayed_customers = (df["months_with_delay"] > 0).sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Default Rate", f"{default_rate:.2f}%")
    col3.metric("Avg Credit Limit", f"{avg_credit_limit:,.0f}")
    col4.metric("Avg Age", f"{avg_age:.1f}")
    col5.metric("Customers With Delay", f"{delayed_customers:,}")

    st.subheader("Default vs Non-Default Customers")

    default_counts = df[TARGET_COL].map({
        0: "No Default",
        1: "Default"
    }).value_counts().reset_index()

    default_counts.columns = ["Status", "Customers"]

    fig = px.pie(
        default_counts,
        names="Status",
        values="Customers",
        title="Default Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Limit Distribution")

    fig = px.histogram(
        df,
        x="limit_bal",
        nbins=40,
        title="Distribution of Credit Limit"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Risk Analysis":
    st.header("Risk Analysis")

    analysis_feature = st.selectbox(
        "Choose a segment to analyze",
        [
            "age_group",
            "limit_bucket",
            "education",
            "marriage",
            "sex",
            "months_with_delay",
            "max_delay"
        ]
    )

    segment_df = (
        df.groupby(analysis_feature)
        .agg(
            total_customers=("id", "count"),
            default_customers=(TARGET_COL, "sum"),
            default_rate=(TARGET_COL, "mean"),
            avg_credit_limit=("limit_bal", "mean"),
            avg_bill_amount=("avg_bill_amt", "mean")
        )
        .reset_index()
    )

    segment_df["default_rate"] = segment_df["default_rate"] * 100
    segment_df = segment_df.sort_values(by="default_rate", ascending=False)

    st.dataframe(segment_df, use_container_width=True)

    fig = px.bar(
        segment_df,
        x=analysis_feature,
        y="default_rate",
        title=f"Default Rate by {analysis_feature}",
        text="default_rate"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delay Behavior vs Default")

    delay_df = (
        df.groupby("months_with_delay")
        .agg(
            total_customers=("id", "count"),
            default_rate=(TARGET_COL, "mean")
        )
        .reset_index()
    )

    delay_df["default_rate"] = delay_df["default_rate"] * 100

    fig = px.line(
        delay_df,
        x="months_with_delay",
        y="default_rate",
        markers=True,
        title="Default Rate by Number of Delayed Payment Months"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model Comparison")

    st.dataframe(comparison_df, use_container_width=True)

    fig = px.bar(
        comparison_df,
        x="model",
        y=["roc_auc", "pr_auc", "f1"],
        barmode="group",
        title="Model Comparison: ROC-AUC vs PR-AUC vs F1"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")

    top_features = importance_df.head(15)

    fig = px.bar(
        top_features.sort_values("importance_mean"),
        x="importance_mean",
        y="feature",
        orientation="h",
        title="Top 15 Important Features"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Threshold Simulator":
    st.header("Threshold Simulator")

    st.write(
        "Change the risk threshold to see how model behavior changes. "
        "A lower threshold catches more risky customers, but also flags more safe customers."
    )

    threshold = st.slider(
        "Default Risk Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    y_true = test_preds["y_true"].values
    y_proba = test_preds["y_proba"].values

    metrics = calculate_threshold_metrics(y_true, y_proba, threshold)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{metrics['precision']:.2f}")
    col2.metric("Recall", f"{metrics['recall']:.2f}")
    col3.metric("F1-score", f"{metrics['f1']:.2f}")
    col4.metric("Customers Flagged", f"{metrics['flagged']:,}")

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ],
        index=["Actual No Default", "Actual Default"],
        columns=["Predicted No Default", "Predicted Default"]
    )

    st.dataframe(cm_df, use_container_width=True)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"At threshold {threshold:.2f}, the model catches "
        f"{metrics['tp']} actual defaulters and misses {metrics['fn']} defaulters."
    )


elif page == "Customer Risk Predictor":
    st.header("Customer Risk Predictor")

    st.write(
        "Enter a customer's profile and payment behavior to estimate default risk."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            limit_bal = st.number_input(
                "Credit Limit",
                min_value=10000,
                max_value=1000000,
                value=200000,
                step=10000
            )

            age = st.number_input(
                "Age",
                min_value=21,
                max_value=80,
                value=35
            )

            sex = st.selectbox(
                "Sex",
                ["male", "female"]
            )

        with col2:
            education = st.selectbox(
                "Education",
                ["graduate_school", "university", "high_school", "others"]
            )

            marriage = st.selectbox(
                "Marriage",
                ["single", "married", "others"]
            )

            current_delay = st.slider(
                "Current Payment Delay",
                min_value=-2,
                max_value=8,
                value=0,
                help="-1 or 0 means no major delay. Higher values mean more delay."
            )

        with col3:
            previous_delay = st.slider(
                "Previous Months Delay",
                min_value=-2,
                max_value=8,
                value=0
            )

            current_bill = st.number_input(
                "Current Bill Amount",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=5000
            )

            current_payment = st.number_input(
                "Current Payment Amount",
                min_value=0,
                max_value=1000000,
                value=10000,
                step=5000
            )

        submitted = st.form_submit_button("Predict Default Risk")

    if submitted:
        sex_code = 1 if sex == "male" else 2

        education_code_map = {
            "graduate_school": 1,
            "university": 2,
            "high_school": 3,
            "others": 4
        }

        marriage_code_map = {
            "married": 1,
            "single": 2,
            "others": 3
        }

        input_data = {
            "id": 1,
            "limit_bal": limit_bal,
            "sex": sex_code,
            "education": education_code_map[education],
            "marriage": marriage_code_map[marriage],
            "age": age,

            "pay_0": current_delay,
            "pay_2": previous_delay,
            "pay_3": previous_delay,
            "pay_4": previous_delay,
            "pay_5": previous_delay,
            "pay_6": previous_delay,

            "bill_amt1": current_bill,
            "bill_amt2": current_bill,
            "bill_amt3": current_bill,
            "bill_amt4": current_bill,
            "bill_amt5": current_bill,
            "bill_amt6": current_bill,

            "pay_amt1": current_payment,
            "pay_amt2": current_payment,
            "pay_amt3": current_payment,
            "pay_amt4": current_payment,
            "pay_amt5": current_payment,
            "pay_amt6": current_payment
        }

        customer_df = pd.DataFrame([input_data])
        customer_clean = clean_credit_data(customer_df)

        feature_columns = metadata["feature_columns"]
        customer_features = customer_clean[feature_columns]

        risk_probability = model.predict_proba(customer_features)[0][1]

        if risk_probability < 0.20:
            risk_band = "Low Risk"
            recommendation = "Customer appears relatively safe."
            st.success(f"Default Risk: {risk_probability:.2%}")
        elif risk_probability < 0.50:
            risk_band = "Medium Risk"
            recommendation = "Manual review recommended before approval."
            st.warning(f"Default Risk: {risk_probability:.2%}")
        else:
            risk_band = "High Risk"
            recommendation = "High-risk customer. Strong manual review required."
            st.error(f"Default Risk: {risk_probability:.2%}")

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Risk Probability", f"{risk_probability:.2%}")
        col2.metric("Risk Band", risk_band)

        st.write("### Recommended Action")
        st.write(recommendation)