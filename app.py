import os
import warnings
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# -------------------------
# Suppress warnings / TF info
# -------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------
# Streamlit Title
# -------------------------
st.title("ModelBench: Classical ML vs Deep Learning")
st.write("Bank Marketing Dataset - UCI Repository")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank-full.csv", sep=";")
    return df

df = load_data()
st.write("Dataset Shape:", df.shape)

# -------------------------
# Preprocessing
# -------------------------
# Encode target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Benchmark Button
# -------------------------
if st.button("Run Benchmark"):

    results = []

    # -------- Logistic Regression --------
    start = time.time()
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_time = time.time() - start

    results.append([
        "Logistic Regression",
        accuracy_score(y_test, lr_preds),
        f1_score(y_test, lr_preds),
        lr_time
    ])

    # -------- Random Forest --------
    start = time.time()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_time = time.time() - start

    results.append([
        "Random Forest",
        accuracy_score(y_test, rf_preds),
        f1_score(y_test, rf_preds),
        rf_time
    ])

    # -------- Neural Network --------
    start = time.time()
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    nn_preds = (model.predict(X_test) > 0.5).astype("int32")
    nn_time = time.time() - start

    results.append([
        "Neural Network",
        accuracy_score(y_test, nn_preds),
        f1_score(y_test, nn_preds),
        nn_time
    ])

    # -------- Results Table --------
    results_df = pd.DataFrame(results,
                              columns=["Model", "Accuracy", "F1 Score", "Training Time (s)"])

    st.subheader("Benchmark Table")
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.2%}",
        "F1 Score": "{:.2%}",
        "Training Time (s)": "{:.2f}"
    }))

    # -------- Accuracy Plot --------
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["Accuracy"], color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy")
    for i, v in enumerate(results_df["Accuracy"]):
        ax.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
    st.pyplot(fig)
