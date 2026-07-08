import streamlit as st

class Info:
    def render(self):
        st.title("👥 About the Team & Project")
 
        st.markdown(
            """
            ### 🎓 Introduction to Machine Learning - Midterm Project
            **Project name:** Student Performance Prediction
            **Dataset:** Student Performance Dataset - UCI Machine Learning Repository
            """
        )
 
        st.divider()
        st.subheader("Team members")
 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                #### Member 1
                - **Name:** Phan Phạm Thùy Vân
                - **Student ID:** 52400254
                - **Role:** Data Preprocessing & Machine Learning
                - Data preprocessing
                - Built and evaluated classification models
                - Selected and saved the best-performing model
                """
            )
        with col2:
            st.markdown(
                """
                #### Member 2
                - **Name:** Hoàng Ngọc Mỹ Anh
                - **Student ID:** 52400041
                -  **Role:** Feature Selection & Data Analysis
                - Correlation analysis
                - Linear Regression and MAE comparison
                - Data visualization and presentation slides
                """
            )
        with col3:
            st.markdown(
                """
                #### Member 3
                - **Name:** Nguyễn Ngọc Kim Lợi
                - **Student ID:** 52400211
                - **Role:** Application & Integration
                - Developed the Streamlit web application
                - Integrated the machine learning model
                - Assembled the project
                """
            )
 
        st.divider()
        st.subheader("Tech stack")
        st.markdown(
            """
            - **Language:** Python > 3.10
            - **Machine Learning:** scikit-learn (Logistic Regression, Random Forest, SVM)
            - **Interface:** Streamlit
            - **Visualization:** Plotly
            - **Data processing:** Pandas, NumPy
            """
        )
 
        st.caption("Project for Introduction to Machine Learning - 2026")
