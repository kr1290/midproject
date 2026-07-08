import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from Train import CATEGORICAL_MAPS


class Statistics:
    """Statistics page split into sub-tabs: Data Statistics, Data Schema,
    Missing Values, Numeric Stats, Heatmap, Numeric Explorer, Categorical."""

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

    # 1. DATA STATISTICS (overview)
    def _render_data_statistics(self):
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

    # 2. DATA SCHEMA
    def _render_data_schema(self):
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

    # 3. MISSING VALUES
    def _render_missing_values(self):
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

    # 4. NUMERIC STATS
    def _render_numeric_stats(self):
        st.subheader("Numeric statistics")
        st.markdown("Descriptive statistics (count, mean, std, min, quartiles, max) for every numerical column.")
        desc = self.df[self.numeric_cols].describe().T
        desc["skew"] = self.df[self.numeric_cols].skew()
        desc = desc.round(2)
        st.dataframe(desc, use_container_width=True)

    # 5. CORRELATION HEATMAP
    def _render_heatmap(self):
        st.subheader("Correlation heatmap")
        enc_df = self._encoded_df()
        corr_matrix = enc_df.drop(columns=["G1", "G2"]).corr(numeric_only=True)
        fig_heatmap = px.imshow(
            corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation heatmap across attributes (excluding G1, G2)",
        )
        fig_heatmap.update_layout(height=700)
        st.plotly_chart(fig_heatmap, use_container_width=True)

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

    # 6. FEATURE SELECTION + LINEAR REGRESSION MAE - Task 2 (Member 2 - Cau2.py)
    def _render_mae_comparison(self):
        st.subheader("Feature selection & MAE comparison (Linear Regression)")
        st.markdown(
            """
            This section compares **Linear Regression** across five correlation-based
            feature sets. Performance is measured using **Mean Absolute Error (MAE)**,
            where lower values indicate better prediction accuracy.
            """
        )

        # --- 1. Correlation with G3 on the full encoded dataset (G1, G2
        # included), same as the original script ---
        enc_df = self._encoded_df()
        target = "G3"
        X_all = enc_df.drop(columns=[target])
        y = enc_df[target]

        corr = enc_df.corr(numeric_only=True)
        corr_target = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)

        # --- 2. Feature sets, same as the original script ---
        feature_sets = {
            "All features (Full)": X_all.columns.tolist(),
            "Top 10 highest correlation": corr_target.head(10).index.tolist(),
            "Top 5 highest correlation": corr_target.head(5).index.tolist(),
            "Threshold |corr| > 0.1": corr_target[corr_target.abs() > 0.1].index.tolist(),
            "Threshold |corr| > 0.2": corr_target[corr_target.abs() > 0.2].index.tolist(),
        }

        # --- 3. Linear Regression + MAE for each feature set ---
        results = []
        for set_name, cols in feature_sets.items():
            if len(cols) == 0:
                continue
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

            mae = mean_absolute_error(y_test, y_pred)
            results.append({"Feature set": set_name, "Number of features": len(cols), "MAE": mae})

        results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

        best_row = results_df.iloc[0]
        st.success(
            f"🏆 Best feature set: **{best_row['Feature set']}** "
            f"(MAE = {best_row['MAE']:.4f}, {int(best_row['Number of features'])} features)"
        )

        st.dataframe(
            results_df.style.format({"MAE": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )

        with st.expander("🔍 View features used in each set"):
            for name, cols in feature_sets.items():
                st.markdown(f"**{name}** ({len(cols)} features): {', '.join(cols) if cols else '—'}")

        # --- 4. Plotly bar chart ---
        fig_mae = px.bar(
            results_df, x="Feature set", y="MAE",
            color="MAE", color_continuous_scale="Greens_r",
            text=results_df["MAE"].map(lambda v: f"{v:.3f}"),
            title="MAE comparison across feature sets (Linear Regression)",
            labels={"MAE": "MAE (lower = better)"},
        )
        fig_mae.update_traces(textposition="outside")
        fig_mae.update_layout(height=550, xaxis_tickangle=-15, showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    # 7. NUMERIC EXPLORER
    def _render_numeric_explorer(self):
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

    # 8. CATEGORICAL EXPLORER
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
            "Data Statistics", "Data Schema", "Heatmap", "MAE Comparison",
            "Missing Values", "Numeric Stats",
            "Numeric Explorer", "Categorical",
        ])
        with tabs[0]:
            self._render_data_statistics()
        with tabs[1]:
            self._render_data_schema()
        with tabs[2]:
            self._render_heatmap()
        with tabs[3]:
            self._render_mae_comparison()
        with tabs[4]:
            self._render_missing_values()
        with tabs[5]:
            self._render_numeric_stats()
        with tabs[6]:
            self._render_numeric_explorer()
        with tabs[7]:
            self._render_categorical_explorer()
