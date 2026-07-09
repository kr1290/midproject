import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from Train import CATEGORICAL_MAPS, DATA_PATH, train_all_classifiers


class Statistics:
    """Statistics page split into sub-tabs: Dataset Overview, Missing Values,
    Numeric Stats, Heatmap, MAE Comparison, Model Comparison, Numeric
    Explorer, Categorical."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(include="object").columns.tolist()
        # add pass/fail column, used for the comparison charts
        self.df_labeled = df.copy()
        self.df_labeled["pass_label"] = np.where(df["G3"] >= 10, "Pass", "Fail")

    def _encoded_df(self) -> pd.DataFrame:
        """Copy of the dataframe with categorical columns encoded the same
        way as Train, so correlation can be computed on all attributes."""
        enc = self.df.copy()
        for col, classes in CATEGORICAL_MAPS.items():
            enc[col] = enc[col].apply(lambda v: classes.index(v) if v in classes else np.nan)
        return enc

    # ---------------------------------------------------------------
    # 1. DATASET OVERVIEW (Data Statistics + Data Schema merged)
    # ---------------------------------------------------------------
    def _render_dataset_overview(self):
        st.subheader("Dataset overview")
        st.markdown(
            f"""
            The **Student Performance Dataset** (UCI Machine Learning Repository)
            collects information on secondary school students from two Portuguese
            schools (Gabriel Pereira - GP and Mousinho da Silveira - MS), including
            demographic, family, study-habit and grade attributes.

            - **Source:** https://archive.ics.uci.edu/dataset/320/student+performance
            - **Rows (Math course file):** {self.df.shape[0]} students
            - **Columns:** {self.df.shape[1]} attributes (numerical + categorical)
            - **Target variable:** **G3** (final grade, range 0–20)
            - **Prediction:** **Pass** if **G3 ≥ 10**, otherwise **Fail**
            """
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", self.df.shape[0])
        c2.metric("Columns", self.df.shape[1])
        pass_rate = (self.df["G3"] >= 10).mean() * 100
        c3.metric("Pass rate", f"{pass_rate:.1f}%")
        c4.metric("Avg. G3", f"{self.df['G3'].mean():.2f}")

        with st.expander("🔍 View sample data (first 5 rows)"):
            st.dataframe(self.df.head())

        st.divider()
        st.subheader("G3 score distribution")
        fig_hist = px.histogram(
            self.df, x="G3", nbins=21, color_discrete_sequence=["#4C78A8"],
            title="Final grade (G3) distribution",
        )
        fig_hist.add_vline(x=10, line_dash="dash", line_color="red",
                            annotation_text="Pass/Fail threshold (10)")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Pass vs Fail count")
        pass_counts = self.df_labeled["pass_label"].value_counts().reset_index()
        pass_counts.columns = ["Result", "Count"]
        fig_pass = px.bar(
            pass_counts, x="Result", y="Count", color="Result",
            color_discrete_map={"Pass": "#2E8B57", "Fail": "#D64545"},
            title="Number of students by Pass/Fail",
        )
        st.plotly_chart(fig_pass, use_container_width=True)

        st.divider()
        st.subheader("Data schema")
        st.markdown("Column name, data type, and value range/options for every attribute in the dataset.")

        schema_rows = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            n_unique = self.df[col].nunique()
            if col in self.categorical_cols:
                col_type = "Categorical"
                values = ", ".join(map(str, sorted(self.df[col].unique())))
            else:
                col_type = "Numerical"
                values = f"{self.df[col].min()} – {self.df[col].max()}"
            schema_rows.append({
                "Column": col,
                "Type": col_type,
                "Dtype": dtype,
                "Unique values": n_unique,
                "Range / Options": values,
            })
        schema_df = pd.DataFrame(schema_rows)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

        st.caption(
            f"**{len(self.numeric_cols)}** numerical columns, "
            f"**{len(self.categorical_cols)}** categorical columns, "
            f"**{self.df.shape[0]}** rows in total."
        )

        st.divider()
        st.subheader("Missing values")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing count": missing.values,
            "Missing %": missing_pct.values,
        }).sort_values("Missing count", ascending=False)

        total_missing = int(missing.sum())
        if total_missing == 0:
            st.success("✅ No missing values found in this dataset - no imputation needed.")
        else:
            st.warning(f"⚠️ Found {total_missing} missing values across the dataset.")

        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        if total_missing > 0:
            fig_missing = px.bar(
                missing_df[missing_df["Missing count"] > 0],
                x="Column", y="Missing count", color="Missing %",
                color_continuous_scale="Reds", title="Missing values per column",
            )
            st.plotly_chart(fig_missing, use_container_width=True)

    # ---------------------------------------------------------------
    # 2. NUMERIC (Numeric Stats + Numeric Explorer merged)
    # ---------------------------------------------------------------
    def _render_numeric(self):
        st.subheader("Numeric statistics")
        st.markdown("Descriptive statistics (count, mean, std, min, quartiles, max) for every numerical column.")
        desc = self.df[self.numeric_cols].describe().T
        desc["skew"] = self.df[self.numeric_cols].skew()
        desc = desc.round(2)
        st.dataframe(desc, use_container_width=True)

        st.divider()
        st.subheader("Numeric explorer")
        col = st.selectbox("Choose a numeric column", self.numeric_cols, key="numeric_explorer_col")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{self.df[col].mean():.2f}")
        c2.metric("Median", f"{self.df[col].median():.2f}")
        c3.metric("Std dev", f"{self.df[col].std():.2f}")
        c4.metric("Min / Max", f"{self.df[col].min()} / {self.df[col].max()}")

        c1, c2 = st.columns(2)
        with c1:
            fig_dist = px.histogram(
                self.df, x=col, nbins=20, color_discrete_sequence=["#4C78A8"],
                title=f"Distribution of {col}",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            fig_box = px.box(
                self.df_labeled, x="pass_label", y=col, color="pass_label",
                color_discrete_map={"Pass": "#2E8B57", "Fail": "#D64545"},
                title=f"{col} by Pass/Fail",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ---------------------------------------------------------------
    # 3. CORRELATION HEATMAP
    # ---------------------------------------------------------------
    def _render_heatmap(self):
        st.subheader("Correlation heatmap")
        enc_df = self._encoded_df()
        corr_matrix = enc_df.drop(columns=["G1", "G2"]).corr(numeric_only=True)

        st.subheader("Correlation of each attribute with G3")
        corr_g3 = corr_matrix["G3"].drop(["G3"]).sort_values(key=lambda s: s.abs(), ascending=False)
        fig_corr_bar = px.bar(
            corr_g3, orientation="h",
            color=corr_g3.values, color_continuous_scale="RdBu_r",
            title="Pearson correlation with G3 (excluding G1, G2)",
            labels={"value": "Correlation coefficient", "index": "Attribute"},
        )
        fig_corr_bar.update_layout(height=600, yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_corr_bar, use_container_width=True)

        # Correlation heatmap - top 10 attributes most correlated with G3
        # (computed on the full encoded dataset, G1/G2 included)
        target = "G3"
        corr_full = enc_df.corr(numeric_only=True)
        corr_target_full = corr_full[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)

        top10_feats = corr_target_full.head(10).index.tolist()
        sub_corr = enc_df[top10_feats + [target]].corr(numeric_only=True)
        fig_top10 = px.imshow(
            sub_corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f",
            title="Correlation heatmap - top 10 attributes most correlated with G3",
        )
        fig_top10.update_layout(height=550)
        st.plotly_chart(fig_top10, use_container_width=True)

        fig_heatmap = px.imshow(
            corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation heatmap across attributes (excluding G1, G2)",
        )
        fig_heatmap.update_layout(height=700)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ---------------------------------------------------------------
    # 5. FEATURE SELECTION + LINEAR REGRESSION MAE - Task 2 (Member 2)
    # ---------------------------------------------------------------
    def _render_mae_comparison(self):
        st.subheader("Feature selection & MAE comparison (Linear Regression)")

        # --- Same preprocessing as the original script (G1, G2 included) ---
        enc_df = self._encoded_df()
        target = "G3"
        X_all = enc_df.drop(columns=[target])
        y = enc_df[target]

        corr = enc_df.corr(numeric_only=True)
        corr_target = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)

        def _mae_for(cols: list[str]) -> float:
            X_sub = X_all[cols]
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            return mean_absolute_error(y_test, y_pred)

        # --- Fixed reference feature sets, same as the original script ---
        feature_sets = {
            "All features (Full)": X_all.columns.tolist(),
            "Top 10 highest correlation": corr_target.head(10).index.tolist(),
            "Top 5 highest correlation": corr_target.head(5).index.tolist(),
            "Threshold |corr| > 0.1": corr_target[corr_target.abs() > 0.1].index.tolist(),
            "Threshold |corr| > 0.2": corr_target[corr_target.abs() > 0.2].index.tolist(),
        }

        results = []
        for set_name, cols in feature_sets.items():
            if len(cols) == 0:
                continue
            mae = _mae_for(cols)
            results.append({"Feature set": set_name, "Number of features": len(cols), "MAE": mae})

        # --- Interactive: let the user pick how many top-correlated
        # features to use, and add that result to the comparison ---
        st.markdown("#### Try a custom number of features")
        max_n = len(X_all.columns)
        n_feats = st.slider(
            "Number of top-correlated features to use",
            min_value=1, max_value=max_n, value=5, key="mae_n_features",
        )
        custom_cols = corr_target.head(n_feats).index.tolist()
        custom_mae = _mae_for(custom_cols)
        st.metric(f"MAE with top {n_feats} feature(s)", f"{custom_mae:.4f}")
        with st.expander("🔍 View the selected features"):
            st.markdown(", ".join(custom_cols))

        # Detailed view: sweep MAE across every possible feature count so
        # the trend (and the point currently picked by the slider) is clear
        sweep_df = pd.DataFrame(
            [{"N features": n, "MAE": _mae_for(corr_target.head(n).index.tolist())}
             for n in range(1, max_n + 1)]
        )
        fig_sweep = px.line(
            sweep_df, x="N features", y="MAE", markers=True,
            title="MAE vs. number of top-correlated features used",
        )
        fig_sweep.add_vline(
            x=n_feats, line_dash="dash", line_color="#ff7f0e",
            annotation_text=f"Selected: {n_feats}", annotation_position="top",
        )
        fig_sweep.add_scatter(
            x=[n_feats], y=[custom_mae], mode="markers",
            marker=dict(size=13, color="#ff7f0e", symbol="star"),
            name=f"Top {n_feats}", showlegend=False,
        )
        best_n = int(sweep_df.loc[sweep_df["MAE"].idxmin(), "N features"])
        best_n_mae = sweep_df["MAE"].min()
        fig_sweep.add_scatter(
            x=[best_n], y=[best_n_mae], mode="markers",
            marker=dict(size=13, color="#2ca02c", symbol="diamond"),
            name=f"Best: Top {best_n}",
        )
        fig_sweep.update_layout(height=450)
        st.plotly_chart(fig_sweep, use_container_width=True)
        st.caption(
            f"📉 The lowest MAE across all possible feature counts is "
            f"**{best_n_mae:.4f}** with the **top {best_n}** correlated features."
        )

        results.append({
            "Feature set": f"Custom (Top {n_feats})",
            "Number of features": n_feats,
            "MAE": custom_mae,
        })

        results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

        best_row = results_df.iloc[0]
        st.success(
            f"🏆 Best feature set so far: **{best_row['Feature set']}** "
            f"(MAE = {best_row['MAE']:.4f}, {int(best_row['Number of features'])} features)"
        )

        st.dataframe(
            results_df.style.format({"MAE": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )

        # --- Bar chart, custom result highlighted in a different color ---
        results_df["Type"] = np.where(
            results_df["Feature set"].str.startswith("Custom"), "Custom", "Reference set"
        )
        fig_mae = px.bar(
            results_df, x="Feature set", y="MAE", color="Type",
            color_discrete_map={"Reference set": "#2ca02c", "Custom": "#ff7f0e"},
            text=results_df["MAE"].map(lambda v: f"{v:.3f}"),
            title="MAE comparison across feature sets (Linear Regression)",
            labels={"MAE": "MAE (lower = better)"},
        )
        fig_mae.update_traces(textposition="outside")
        fig_mae.update_layout(height=550, xaxis_tickangle=-15)
        st.plotly_chart(fig_mae, use_container_width=True)

    # ---------------------------------------------------------------
    # 6. CLASSIFICATION MODEL COMPARISON - Task 1 (Member 1)
    # ---------------------------------------------------------------
    def _render_model_comparison(self):
        st.subheader("Classification model comparison")
        st.markdown(
            """
            Compares the performance of **Logistic Regression**, **Random Forest**, and **SVM**
            using the same 80/20 train-test split (`random_state=42`) to predict
            student **Pass/Fail** (`G3 ≥ 10`). Performance is evaluated using
            **Accuracy, Precision, Recall, F1-score, training time, testing time,
            and confusion matrices**.
            """
        )

        _, _, metrics_df, confusions = train_all_classifiers(str(DATA_PATH))

        best_row = metrics_df.iloc[0]
        st.success(
            f"🏆 Best model: **{best_row['Model']}** "
            f"(Accuracy = {best_row['Accuracy']:.4f})"
        )

        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                "F1-score": "{:.4f}", "Train time (s)": "{:.4f}", "Test time (s)": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        metrics_long = metrics_df.melt(
            id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
            var_name="Metric", value_name="Score",
        )
        fig_models = px.bar(
            metrics_long, x="Model", y="Score", color="Metric", barmode="group",
            text=metrics_long["Score"].map(lambda v: f"{v:.3f}"),
            title="Model performance comparison (Accuracy / Precision / Recall / F1-score)",
        )
        fig_models.update_traces(textposition="outside")
        fig_models.update_layout(height=550, yaxis_range=[0, 1])
        st.plotly_chart(fig_models, use_container_width=True)

        fig_time = px.bar(
            metrics_df, x="Model", y=["Train time (s)", "Test time (s)"], barmode="group",
            title="Training / inference time by model",
            labels={"value": "Time (seconds)", "variable": "Phase"},
        )
        fig_time.update_layout(height=450)
        st.plotly_chart(fig_time, use_container_width=True)

        # --- Confusion matrices: shows exactly where each model gets
        # Pass/Fail wrong, side by side for direct comparison ---
        st.markdown("#### Confusion matrices (test set)")
        cols = st.columns(len(confusions))
        for col, (name, cm) in zip(cols, confusions.items()):
            with col:
                fig_cm = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Pred: Fail", "Pred: Pass"], y=["Actual: Fail", "Actual: Pass"],
                    title=name,
                )
                fig_cm.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)

    # ---------------------------------------------------------------
    # 7. CATEGORICAL EXPLORER
    # ---------------------------------------------------------------
    def _render_categorical_explorer(self):
        st.subheader("Categorical explorer")
        col = st.selectbox("Choose a categorical column", self.categorical_cols, key="categorical_explorer_col")

        c1, c2 = st.columns(2)
        with c1:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, "Count"]
            fig_count = px.bar(
                counts, x=col, y="Count", color=col,
                title=f"Count of students by {col}",
            )
            st.plotly_chart(fig_count, use_container_width=True)
        with c2:
            pass_rate = (
                self.df_labeled.groupby(col)["pass_label"]
                .apply(lambda s: (s == "Pass").mean() * 100)
                .reset_index(name="Pass rate (%)")
            )
            fig_rate = px.bar(
                pass_rate, x=col, y="Pass rate (%)", color=col,
                title=f"Pass rate (%) by {col}",
            )
            st.plotly_chart(fig_rate, use_container_width=True)

    def render(self):
        st.title("📊 Data Statistics")

        tabs = st.tabs([
            "Dataset Overview", "Numeric", "Categorical",
            "Model Comparison", "Heatmap", "MAE Comparison",
        ])
        with tabs[0]:
            self._render_dataset_overview()
        with tabs[1]:
            self._render_numeric()
        with tabs[2]:
            self._render_categorical_explorer()
        with tabs[3]:
            self._render_model_comparison()
        with tabs[4]:
            self._render_heatmap()
        with tabs[5]:
            self._render_mae_comparison()
