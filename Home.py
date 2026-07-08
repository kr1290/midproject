import pandas as pd
import streamlit as st

from Train import CATEGORICAL_MAPS, NUMERICAL_RANGES, LABELS, Train


class Home:
    STUDYTIME_OPTIONS = {
        "< 2 hours/week": 1,
        "2 - 5 hours/week": 2,
        "5 - 10 hours/week": 3,
        "> 10 hours/week": 4,
    }
    
    TRAVELTIME_OPTIONS = {
        "< 15 minutes": 1,
        "15 - 30 minutes": 2,
        "30 minutes - 1 hour": 3,
        "> 1 hour": 4,
    }

    QUALITY_OPTIONS = {
        "Very poor": 1,
        "Poor": 2,
        "Average": 3,
        "Good": 4,
        "Excellent": 5,
    }

    GOOUT_OPTIONS = {
        "Very rarely": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Often": 4,
        "Very often": 5,
    }

    ALCOHOL_OPTIONS = {
        "Very low": 1,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very high": 5,
    }
    PRIMARY_FIELDS = ["age", "sex", "studytime", "failures", "absences", "schoolsup", "higher", "internet"]
 
    ADVANCED_CATEGORICAL = ["school", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                             "reason", "guardian", "famsup", "paid", "activities",
                             "nursery", "romantic"]
    ADVANCED_NUMERICAL = ["Medu", "Fedu", "traveltime", "famrel", "freetime",
                           "goout", "Dalc", "Walc", "health"]
 
    def __init__(self, df: pd.DataFrame, train: Train):
        self.df = df
        self.train = train
        self.default_num = {c: int(df[c].median()) for c in NUMERICAL_RANGES}
        self.default_cat = {c: df[c].mode()[0] for c in CATEGORICAL_MAPS}
 
    def _field(self, input_data: dict, col: str):

    # Weekly study time
        if col == "studytime":

            options = list(self.STUDYTIME_OPTIONS.keys())

            default = list(self.STUDYTIME_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Weekly study time",
                options,
                index=default,
            )

            input_data[col] = self.STUDYTIME_OPTIONS[choice]

        # Travel time
        elif col == "traveltime":

            options = list(self.TRAVELTIME_OPTIONS.keys())

            default = list(self.TRAVELTIME_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Travel time to school",
                options,
                index=default,
            )

            input_data[col] = self.TRAVELTIME_OPTIONS[choice]

        # Family relationship
        elif col == "famrel":

            options = list(self.QUALITY_OPTIONS.keys())

            default = list(self.QUALITY_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Family relationship quality",
                options,
                index=default,
            )

            input_data[col] = self.QUALITY_OPTIONS[choice]

        # Free time
        elif col == "freetime":

            options = list(self.QUALITY_OPTIONS.keys())

            default = list(self.QUALITY_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Free time after school",
                options,
                index=default,
            )

            input_data[col] = self.QUALITY_OPTIONS[choice]

        # Health
        elif col == "health":

            options = list(self.QUALITY_OPTIONS.keys())

            default = list(self.QUALITY_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Health status",
                options,
                index=default,
            )

            input_data[col] = self.QUALITY_OPTIONS[choice]

        # Going out
        elif col == "goout":

            options = list(self.GOOUT_OPTIONS.keys())

            default = list(self.GOOUT_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Going out with friends",
                options,
                index=default,
            )

            input_data[col] = self.GOOUT_OPTIONS[choice]

        # Workday alcohol
        elif col == "Dalc":

            options = list(self.ALCOHOL_OPTIONS.keys())

            default = list(self.ALCOHOL_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Workday alcohol use",
                options,
                index=default,
            )

            input_data[col] = self.ALCOHOL_OPTIONS[choice]

        # Weekend alcohol
        elif col == "Walc":

            options = list(self.ALCOHOL_OPTIONS.keys())

            default = list(self.ALCOHOL_OPTIONS.values()).index(
                self.default_num[col]
            )

            choice = st.selectbox(
                "Weekend alcohol use",
                options,
                index=default,
            )

            input_data[col] = self.ALCOHOL_OPTIONS[choice]

        # Categorical
        elif col in CATEGORICAL_MAPS:

            options = CATEGORICAL_MAPS[col]

            default = options.index(self.default_cat[col])

            input_data[col] = st.selectbox(
                LABELS[col],
                options,
                index=default,
            )

        # Numeric
        else:

            lo, hi = NUMERICAL_RANGES[col]

            input_data[col] = st.number_input(
                LABELS[col],
                min_value=lo,
                max_value=hi,
                value=self.default_num[col],
                step=1,
            )

    def _render_form(self) -> tuple[dict, bool]:
        input_data = {}
        with st.form("prediction_form"):
            st.subheader("Input information")
            cols = st.columns(4)
            for i, col in enumerate(self.PRIMARY_FIELDS):
                with cols[i % 4]:
                    self._field(input_data, col)
 
            st.caption(
                "💡 Advanced options (optional) — if left unchanged, the app uses "
                "the most common values from the training dataset."
            )
            with st.expander("Advanced options (optional)"):
                adv_cols = self.ADVANCED_CATEGORICAL + self.ADVANCED_NUMERICAL
                cols = st.columns(4)
                for i, col in enumerate(adv_cols):
                    with cols[i % 4]:
                        self._field(input_data, col)
 
            submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")
 
        return input_data, submitted
 
    def _render_result(self, pred: int, proba):
        st.divider()
        st.subheader("Prediction result")
        if pred == 1:
            msg = "### ✅ Prediction: PASS"
            if proba is not None:
                msg += f"\nProbability of passing: **{proba[1] * 100:.1f}%**"
            st.success(msg)
        else:
            msg = "### ❌ Prediction: FAIL"
            if proba is not None:
                msg += f"\nProbability of failing: **{proba[0] * 100:.1f}%**"
            st.error(msg)
        st.caption("Model in use: **Logistic Regression**")
 
    def render(self):
        title_col, badge_col = st.columns([5, 1])
        with title_col:
            st.title("🎓 Student Performance Predictor")
        with badge_col:
            st.markdown(
                """
                <div style="text-align:right; margin-top:32px;">
                    <span style="
                        background:#000080;
                        color:white;
                        padding:6px 14px;
                        border-radius:20px;
                        font-size:13px;
                        font-weight:600;
                        white-space: nowrap;
                        display:inline-block;">
                        Logistic Regression
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            "Enter a student's information to predict whether the student will **Pass or Fail**."
        )
        st.divider()
 
        input_data, submitted = self._render_form()
        if submitted:
            pred, proba = self.train.predict(input_data)
            self._render_result(pred, proba)