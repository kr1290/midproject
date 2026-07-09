import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR.parent / "DATA" / "student-mat.csv"

MODEL_PATH = BASE_DIR / "best_student_model.pkl"

SCALER_PATH = BASE_DIR / "scaler.pkl"

# (X = df.drop(columns=['G1','G2','G3','pass']))
FEATURE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]


CATEGORICAL_MAPS = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["R", "U"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["A", "T"],
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["father", "mother", "other"],
    "schoolsup": ["no", "yes"],
    "famsup": ["no", "yes"],
    "paid": ["no", "yes"],
    "activities": ["no", "yes"],
    "nursery": ["no", "yes"],
    "higher": ["no", "yes"],
    "internet": ["no", "yes"],
    "romantic": ["no", "yes"],
}

NUMERICAL_RANGES = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
    "studytime": (1, 4), "failures": (0, 4), "famrel": (1, 5), "freetime": (1, 5),
    "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 75),
}

LABELS = {
    "school": "School", "sex": "Sex", "age": "Age",
    "address": "Address type", "famsize": "Family size",
    "Pstatus": "Parents' cohabitation", "Medu": "Mother's education (0-4)",
    "Fedu": "Father's education (0-4)", "Mjob": "Mother's job",
    "Fjob": "Father's job", "reason": "Reason for choosing school",
    "guardian": "Guardian", "traveltime": "Travel time to school (1-4)",
    "studytime": "Weekly study time (1-4)", "failures": "Past class failures",
    "schoolsup": "Extra educational support", "famsup": "Family educational support",
    "paid": "Extra paid classes", "activities": "Extracurricular activities",
    "nursery": "Attended nursery school", "higher": "Wants higher education",
    "internet": "Internet access at home", "romantic": "In a romantic relationship",
    "famrel": "Family relationship quality (1-5)", "freetime": "Free time after school (1-5)",
    "goout": "Going out with friends (1-5)", "Dalc": "Workday alcohol use (1-5)",
    "Walc": "Weekend alcohol use (1-5)", "health": "Health status (1-5)",
    "absences": "Number of absences",
}

# Names of the 3 classifiers compared in Task 1 (motel_training.py)
CLASSIFIER_NAMES = ["Logistic Regression", "Random Forest", "SVM"]


# MODULE-LEVEL CACHED LOADERS
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


@st.cache_resource
def load_model_and_scaler(model_path: str, scaler_path: str):
    return joblib.load(model_path), joblib.load(scaler_path)


@st.cache_resource
def train_all_classifiers(data_path: str):
    """Reproduces Member 1's Task 1 script (motel_training.py) exactly:
    drop missing values, build the pass/fail target from G3, drop
    G1/G2/G3/pass from the features, Label-encode text columns, split
    80/20 with random_state=42, scale, then train & evaluate Logistic
    Regression, Random Forest and SVM on the same split.

    Returns (trained_models: dict, scaler, metrics_df, confusions: dict) -
    all cached so the (cheap) retraining only happens once per session.
    confusions maps model name -> 2x2 confusion matrix (rows=actual,
    columns=predicted, order [Fail, Pass]).
    """
    df = pd.read_csv(data_path, sep=";")
    df = df.dropna()
    df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

    X = df.drop(columns=["G1", "G2", "G3", "pass"])
    y = df["pass"]

    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # probability=True on SVC only adds Platt-scaled probabilities on top -
    # it does not change predictions/accuracy, but lets the app show a
    # confidence score for SVM too.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True),
    }

    trained = {}
    rows = []
    confusions = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_s, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test_s)
        test_time = time.time() - t0

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Train time (s)": train_time,
            "Test time (s)": test_time,
        })
        trained[name] = model
        confusions[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return trained, scaler, metrics_df, confusions


class Train:
    """Wraps the 3 classification models (Logistic Regression, Random
    Forest, SVM) trained on Task 1's pass/fail target. Supports predicting
    with a chosen model, or with all 3 at once for comparison."""

    def __init__(self, data_path: str = DATA_PATH):
        self.models, self.scaler, self.metrics_df, self.confusions = train_all_classifiers(str(data_path))

    @staticmethod
    def encode_input(input_data: dict) -> pd.DataFrame:
        """Turn a dict of raw, human-readable student info into a single-row
        DataFrame with 30 columns, in the same order/encoding as training."""
        row = {}
        for col in FEATURE_COLUMNS:
            value = input_data[col]
            if col in CATEGORICAL_MAPS:
                row[col] = CATEGORICAL_MAPS[col].index(value)
            else:
                row[col] = value
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, input_data: dict, model_name: str = "Logistic Regression"):
        """Returns (predicted label 0/1, probability array [P(fail), P(pass)])
        using the chosen model."""
        X = self.encode_input(input_data)
        X_scaled = self.scaler.transform(X)
        model = self.models[model_name]
        pred = int(model.predict(X_scaled)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
        return pred, proba

    def predict_all(self, input_data: dict) -> dict:
        """Returns {model_name: (pred, proba)} for every trained model, used
        to build the Home page's model comparison section."""
        X = self.encode_input(input_data)
        X_scaled = self.scaler.transform(X)
        results = {}
        for name, model in self.models.items():
            pred = int(model.predict(X_scaled)[0])
            proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
            results[name] = (pred, proba)
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            print(name)
            print(proba)
        return results
