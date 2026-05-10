# Loan Default Risk Prediction & Analysis Dashboard

An end-to-end machine learning and analytics project that predicts the probability of credit card payment default using customer financial behavior, repayment history, and demographic features.

The project includes data cleaning, SQL-based analysis, machine learning model comparison, threshold tuning, feature importance analysis, and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

Streamlit App: https://loan-default-risk-dashboard.streamlit.app/  

---

## 📌 Project Overview

Credit default prediction is an important problem in financial analytics. Banks and lenders need to identify customers who may fail to repay on time so that they can take better risk-management decisions.

This project predicts whether a customer is likely to default on next month’s credit card payment.

The dashboard helps answer:

- Which customer segments have higher default risk?
- Which features are most important for predicting default?
- How do different risk thresholds affect business decisions?
- What is the predicted default risk for a new customer?

---

## 🧠 Problem Statement

The goal is to build a machine learning system that can classify customers as default-risk or non-default-risk based on their credit limit, bill amounts, payment amounts, repayment delays, age, education, marital status, and other customer attributes.

This is a binary classification problem:

```text
0 = No Default
1 = Default
