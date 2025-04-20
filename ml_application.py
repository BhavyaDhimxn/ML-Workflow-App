import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error, classification_report
)
import joblib
import warnings
import streamlit as st
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")


class MLApplication:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        self.target_column = None
        self.problem_type = None
        self.numerical_cols = None
        self.categorical_cols = None

    def load_data(self, uploaded_file, file_extension):
        try:
            if file_extension == 'csv':
                self.data = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                self.data = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                self.data = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return False
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False

    def explore_data(self):
        if self.data is not None:
            st.subheader("Data Summary")
            st.write(self.data.describe())
            st.write("Missing Values:")
            st.write(self.data.isnull().sum())
            st.write("Data Types:")
            st.write(self.data.dtypes)
            st.write("Correlation Matrix:")
            st.dataframe(self.data.corr(numeric_only=True))
            st.write("Sample Rows:")
            st.dataframe(self.data.head())
        else:
            st.warning("Load data first.")

    def preprocess_data(self):
        if self.data is None:
            st.warning("Load data first.")
            return

        self.target_column = st.selectbox("Select Target Column", self.data.columns)
        if self.target_column is None:
            return

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        st.write("Numerical Features:", self.numerical_cols)
        st.write("Categorical Features:", self.categorical_cols)

        self.problem_type = "regression" if y.dtype.kind in "fc" else "classification"
        st.success(f"Problem Type: {self.problem_type.title()}")

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, self.numerical_cols),
            ("cat", cat_pipeline, self.categorical_cols)
        ])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        st.success("Preprocessing pipeline created and data split.")

    def feature_engineering(self):
        if self.preprocessor is None:
            st.warning("Run preprocessing step first.")
            return

        apply_pca = st.checkbox("Apply PCA for Dimensionality Reduction")
        if apply_pca:
            n_components = st.slider("Number of Components", min_value=1, max_value=20, value=5)
            self.preprocessor = Pipeline(steps=[
                ("pre", self.preprocessor),
                ("pca", PCA(n_components=n_components))
            ])
            st.success(f"PCA applied with {n_components} components.")
        else:
            st.info("PCA not applied.")

    def train_model(self):
        if self.preprocessor is None:
            st.warning("Run preprocessing step first.")
            return

        model_choice = None
        if self.problem_type == "classification":
            model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
        else:
            model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression"])

        if model_choice:
            if self.problem_type == "classification":
                if model_choice == "Random Forest":
                    model = RandomForestClassifier()
                else:
                    model = LogisticRegression()
            else:
                if model_choice == "Random Forest":
                    model = RandomForestRegressor()
                else:
                    model = LinearRegression()

            pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("model", model)
            ])

            pipeline.fit(self.X_train, self.y_train)
            self.model = pipeline

            st.success(f"{model_choice} model trained successfully!")

    def evaluate_model(self):
        if self.model is None:
            st.warning("Train the model first.")
            return

        y_pred = self.model.predict(self.X_test)

        if self.problem_type == "classification":
            st.subheader("Classification Metrics")
            st.write("Accuracy:", accuracy_score(self.y_test, y_pred))
            st.write("Precision:", precision_score(self.y_test, y_pred, average='macro'))
            st.write("Recall:", recall_score(self.y_test, y_pred, average='macro'))
            st.write("F1 Score:", f1_score(self.y_test, y_pred, average='macro'))
            st.text("Classification Report:")
            st.text(classification_report(self.y_test, y_pred))
        else:
            st.subheader("Regression Metrics")
            st.write("R² Score:", r2_score(self.y_test, y_pred))
            st.write("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
            st.write("Mean Absolute Error:", mean_absolute_error(self.y_test, y_pred))

    def save_model(self):
        if self.model is not None:
            filename = st.text_input("Enter filename to save model", value="trained_model.pkl")
            if st.button("Save Model"):
                joblib.dump(self.model, filename)
                st.success(f"Model saved as {filename}")
        else:
            st.warning("Train a model before saving.")


def main():
    st.set_page_config(page_title="ML Workflow App", layout="wide")
    st.title("🚀 Machine Learning Workflow App")
    st.caption("Upload dataset, preprocess, train, evaluate, and save your model easily.")

    if 'ml_app' not in st.session_state:
        st.session_state.ml_app = MLApplication()

    with st.sidebar:
        selected = option_menu(
            "Steps",
            ["Data Loading", "Data Exploration", "Data Preprocessing",
             "Feature Engineering", "Model Training", "Model Evaluation", "Save Model"],
            icons=['upload', 'bar-chart', 'sliders', 'layers', 'robot', 'graph-up', 'download'],
            default_index=0
        )

    app = st.session_state.ml_app

    if selected == "Data Loading":
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file",
                                         type=['csv', 'xlsx', 'xls', 'json'])
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if app.load_data(uploaded_file, file_ext):
                st.success("✅ Data Loaded Successfully!")
                st.dataframe(app.data.head())

    elif selected == "Data Exploration":
        app.explore_data()

    elif selected == "Data Preprocessing":
        app.preprocess_data()

    elif selected == "Feature Engineering":
        app.feature_engineering()

    elif selected == "Model Training":
        app.train_model()

    elif selected == "Model Evaluation":
        app.evaluate_model()

    elif selected == "Save Model":
        app.save_model()


if __name__ == "__main__":
    main()
