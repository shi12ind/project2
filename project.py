import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("Bankruptcy Prediction Application")
st.write("""
This application allows users to upload a dataset, perform exploratory data analysis, 
and build machine learning models to predict bankruptcy.
""")

# File Upload
uploaded_file = st.file_uploader("Upload the Bankruptcy Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_excel(uploaded_file)
    
    # Data Overview
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.write("Shape of the dataset:", data.shape)
    st.write("Data types:")
    st.write(data.dtypes)

    # Encode target variable
    if 'credibility' in data.columns:
        st.write("Encoding the target variable `credibility`.")
        label_encoder = LabelEncoder()
        data['credibility'] = label_encoder.fit_transform(data['credibility'])
    else:
        st.error("Target column `credibility` not found!")
    
    # Feature and target separation
    X = data.drop('credibility', axis=1)
    y = data['credibility']
    
    # Data Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    
    # Logistic Regression
    st.subheader("Logistic Regression")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

    st.write("Logistic Regression Metrics:")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_lr))
    st.write("Precision:", precision_score(y_test, y_pred_lr))
    st.write("Recall:", recall_score(y_test, y_pred_lr))
    st.write("F1 Score:", f1_score(y_test, y_pred_lr))
    st.write("AUC-ROC:", roc_auc_score(y_test, y_prob_lr))

    # Decision Tree Classifier
    st.subheader("Decision Tree Classifier")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

    st.write("Decision Tree Metrics:")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
    st.write("Precision:", precision_score(y_test, y_pred_dt))
    st.write("Recall:", recall_score(y_test, y_pred_dt))
    st.write("F1 Score:", f1_score(y_test, y_pred_dt))
    st.write("AUC-ROC:", roc_auc_score(y_test, y_prob_dt))

    # Random Forest Classifier
    st.subheader("Random Forest Classifier")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    st.write("Random Forest Metrics:")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
    st.write("Precision:", precision_score(y_test, y_pred_rf))
    st.write("Recall:", recall_score(y_test, y_pred_rf))
    st.write("F1 Score:", f1_score(y_test, y_pred_rf))
    st.write("AUC-ROC:", roc_auc_score(y_test, y_prob_rf))

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix Visualization")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(pd.crosstab(y_test, y_pred_lr), annot=True, fmt='d', ax=ax[0], cmap='Blues')
    ax[0].set_title("Logistic Regression")
    sns.heatmap(pd.crosstab(y_test, y_pred_dt), annot=True, fmt='d', ax=ax[1], cmap='Greens')
    ax[1].set_title("Decision Tree")
    sns.heatmap(pd.crosstab(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[2], cmap='Oranges')
    ax[2].set_title("Random Forest")
    st.pyplot(fig)
