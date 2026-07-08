import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "DATA" / "student-mat.csv"

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


# MODULE-LEVEL CACHED LOADERS 
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


@st.cache_resource
def load_model_and_scaler(model_path: str, scaler_path: str):
    return joblib.load(model_path), joblib.load(scaler_path)


class Train:
    """Wraps the classification model (.pkl) and scaler (.pkl) trained by
    Member 1 (Task 1). Only responsible for encoding input in the right
    format and calling model.predict / predict_proba - no retraining here."""

    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.model, self.scaler = load_model_and_scaler(model_path, scaler_path)

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

    def predict(self, input_data: dict):
        """Returns (predicted label 0/1, probability array [P(fail), P(pass)])."""
        X = self.encode_input(input_data)
        X_scaled = self.scaler.transform(X)
        pred = int(self.model.predict(X_scaled)[0])
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)[0]
        return pred, proba
