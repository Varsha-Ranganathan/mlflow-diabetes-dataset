import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import mlflow
import mlflow.sklearn
from datetime import datetime
from math import sqrt

# Set the MLFlow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load data
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df

# Transform data
def data_transform(df):
    df.columns = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target"]
    return df

# Train model and return metrics
def train_model(df, model_type, **kwargs):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Linear Regression":
        model = LinearRegression(**kwargs)
    elif model_type == "Ridge Regression":
        model = Ridge(**kwargs)
    elif model_type == "Lasso Regression":
        model = Lasso(**kwargs)
    elif model_type == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = GridSearchCV(RandomForestRegressor(**kwargs), param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(kwargs)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
        
        run_name = run.info.run_name
        run_id = run.info.run_id
        
        return model, mse, mae, r2, run_name, run_id

# Save model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load model
def load_model(file):
    model = pickle.load(file)
    return model

# Make predictions
def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions

# Get the best model based on the lowest RMSE
def get_best_model():
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])
    best_rmse = float('inf')
    best_model = None
    for run in runs:
        mse = run.data.metrics.get("mse", None)
        if mse is not None:
            rmse = sqrt(mse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = run
    return best_model, best_rmse

# Get all models and their metrics
def get_all_models():
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])
    models = []
    for run in runs:
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = run.data.params["model_type"]
        mse = run.data.metrics.get("mse", None)
        mae = run.data.metrics.get("mae", None)
        r2 = run.data.metrics.get("r2", None)
        rmse = sqrt(mse) if mse is not None else None
        run_name = run.info.run_name
        run_id = run.info.run_id
        run_url = f"http://localhost:5000/#/experiments/0/runs/{run_id}"
        models.append({"run_name": run_name, "model_uri": model_uri, "model_name": model_name, "rmse": rmse, "mae": mae, "r2": r2, "run_url": run_url})
    return models

# Streamlit app
st.title("Diabetes Model Training and Prediction")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Train Model", "View and Download Models", "Register Model", "Predictions"])

with tab1:
    st.header("Train Model")

    # Load and transform data
    df_raw = load_data()
    df_transformed = data_transform(df_raw)

    # Display data
    st.subheader("Transformed Diabetes Dataset")
    st.write(df_transformed.head())

    # Model selection
    model_type = st.selectbox("Select Model", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

    # Model parameters
    if model_type == "Ridge Regression" or model_type == "Lasso Regression":
        alpha = st.slider("Alpha", 0.01, 10.0)
        model_params = {"alpha": alpha}
    elif model_type == "Random Forest":
        n_estimators = st.slider("Number of Estimators", 100, 300, step=100)
        max_depth = st.slider("Max Depth", 10, 30, step=10)
        min_samples_split = st.slider("Min Samples Split", 2, 10, step=1)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 4, step=1)
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
    else:
        model_params = {}

    # Train model and display metrics
    if st.button("Train Model"):
        model, mse, mae, r2, run_name, run_id = train_model(df_transformed, model_type, **model_params)
        st.success(f"{model_type} model trained successfully!")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R-squared: {r2}")

        # Save model and provide download button
        save_model(model, "diabetes_model.pkl")
        with open("diabetes_model.pkl", "rb") as file:
            st.download_button(label="Download Model", data=file, file_name="diabetes_model.pkl")

        # Suggest the best model
        best_model, best_rmse = get_best_model()
        if best_model:
            st.write(f"The best model so far has an RMSE of {best_rmse}")
        else:
            st.write("No models have been logged yet.")

with tab2:
    st.header("View and Download Models")

    # Display all models and their metrics
    models = get_all_models()
    if models:
        st.subheader("Logged Models")
        df_models = pd.DataFrame(models)
        st.dataframe(df_models)
        selected_run_name = st.selectbox("Select Run Name to Download", df_models["run_name"])
        if selected_run_name:
            selected_model_uri = df_models.loc[df_models['run_name'] == selected_run_name, 'model_uri'].values[0]
            model = mlflow.sklearn.load_model(selected_model_uri)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{df_models.loc[df_models['run_name'] == selected_run_name, 'model_name'].values[0]}_{timestamp}.pkl"
            save_model(model, model_name)
            with open(model_name, "rb") as file:
                st.download_button(label="Download Model", data=file, file_name=model_name)
            st.write(f"RMSE: {df_models.loc[df_models['run_name'] == selected_run_name, 'rmse'].values[0]}")
            st.write(f"Run URL: {df_models.loc[df_models['run_name'] == selected_run_name, 'run_url'].values[0]}")
    else:
        st.write("No models have been logged yet.")

with tab3:
    st.header("Register Model")

    # Display all models and their metrics
    models = get_all_models()
    if models:
        st.subheader("Logged Models")
        df_models = pd.DataFrame(models)
        st.dataframe(df_models)
        selected_run_name = st.selectbox("Select Run Name to Register", df_models["run_name"])
        if selected_run_name:
            selected_model_uri = df_models.loc[df_models['run_name'] == selected_run_name, 'model_uri'].values[0]
            st.write(f"RMSE: {df_models.loc[df_models['run_name'] == selected_run_name, 'rmse'].values[0]}")
            st.write(f"Run URL: {df_models.loc[df_models['run_name'] == selected_run_name, 'run_url'].values[0]}")
            
            # Input for model name
            model_name = st.text_input("Enter Model Name for Registration")
            
            # Register the model
            if st.button("Register Model"):
                if model_name:
                    try:
                        mlflow.register_model(model_uri=selected_model_uri, name=model_name)
                        st.success(f"Model '{model_name}' registered successfully!")
                    except Exception as e:
                        st.error(f"An error occurred while registering the model: {e}")
                else:
                    st.warning("Please enter a model name for registration.")
    else:
        st.write("No models have been logged yet.")

    # Display previously registered models
    st.subheader("Previously Registered Models")
    client = mlflow.tracking.MlflowClient()
    registered_models = client.search_registered_models()
    if registered_models:
        df_registered_models = pd.DataFrame([{
            "name": model.name,
            "latest_version": model.latest_versions[0].version,
            "creation_timestamp": datetime.fromtimestamp(model.latest_versions[0].creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            "description": model.latest_versions[0].description,
            "run_id": model.latest_versions[0].run_id
        } for model in registered_models])
        st.dataframe(df_registered_models)
        
        selected_model_name = st.selectbox("Select Registered Model to View Artifacts", df_registered_models["name"])
        if selected_model_name:
            selected_model_version = df_registered_models.loc[df_registered_models['name'] == selected_model_name, 'latest_version'].values[0]
            selected_model_run_id = df_registered_models.loc[df_registered_models['name'] == selected_model_name, 'run_id'].values[0]
            st.write(f"Model Name: {selected_model_name}")
            st.write(f"Version: {selected_model_version}")
            st.write(f"Run ID: {selected_model_run_id}")
            
            # Display artifacts related to model usage
            artifacts = client.list_artifacts(selected_model_run_id)
            if artifacts:
                st.subheader("Artifacts")
                for artifact in artifacts:
                    artifact_path = client.download_artifacts(selected_model_run_id, artifact.path)
                    st.write(f"Artifact Path: {artifact.path}")
                    st.write(f"Artifact Size: {artifact.file_size} bytes")
                    st.write(f"Downloaded to: {artifact_path}")
                    
                    # Display content of the artifact if it's a text file
                    if artifact.path.endswith('.txt') or artifact.path.endswith('.md'):
                        with open(artifact_path, 'r') as file:
                            content = file.read()
                            st.text_area(f"Content of {artifact.path}", content, height=200)
                    elif artifact.path.endswith('.py'):
                        with open(artifact_path, 'r') as file:
                            content = file.read()
                            st.code(content, language='python')
            else:
                st.write("No artifacts found for this model.")
    else:
        st.write("No models have been registered yet.")

with tab4:
    st.header("Predictions")

    # Main choice buttons
    choice = st.radio("You can choose to uplaod a previously trained model OR choose form one of the registered models:", ("Upload Model", "Select Registered Model"))

    if choice == "Upload Model":
        st.subheader("Upload Model")
        uploaded_file = st.file_uploader("Upload Model", type="pkl")
        if uploaded_file is not None:
            model = load_model(uploaded_file)
            st.success("Model uploaded successfully!")

            # Make predictions with uploaded model
            st.subheader("Make Predictions with Uploaded Model")
            input_data = st.text_area("Enter data for prediction (comma-separated values)")
            if st.button("Predict with Uploaded Model"):
                input_data = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
                predictions = make_predictions(model, input_data)
                st.write("Predictions:", predictions)

    elif choice == "Select Registered Model":
        st.subheader("Select Registered Model")
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        if registered_models:
            df_registered_models = pd.DataFrame([{
                "name": model.name,
                "latest_version": model.latest_versions[0].version,
                "creation_timestamp": datetime.fromtimestamp(model.latest_versions[0].creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "description": model.latest_versions[0].description,
                "run_id": model.latest_versions[0].run_id
            } for model in registered_models])
            st.dataframe(df_registered_models)
            
            selected_model_name = st.selectbox("Select Registered Model to Make Predictions", df_registered_models["name"])
            if selected_model_name:
                selected_model_version = df_registered_models.loc[df_registered_models['name'] == selected_model_name, 'latest_version'].values[0]
                selected_model_run_id = df_registered_models.loc[df_registered_models['name'] == selected_model_name, 'run_id'].values[0]
                st.write(f"Model Name: {selected_model_name}")
                st.write(f"Version: {selected_model_version}")
                st.write(f"Run ID: {selected_model_run_id}")
                
                # Load selected registered model
                selected_model_uri = f"models:/{selected_model_name}/{selected_model_version}"
                model = mlflow.sklearn.load_model(selected_model_uri)
                st.success(f"Registered model '{selected_model_name}' loaded successfully!")

                # Make predictions with selected registered model
                st.subheader("Make Predictions with Registered Model")
                input_data = st.text_area("Enter data for prediction (comma-separated values)", key="registered_model_input")
                if st.button("Predict with Registered Model"):
                    input_data = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
                    predictions = make_predictions(model, input_data)
                    st.write("Predictions:", predictions)
        else:
            st.write("No models have been registered yet.")