# mlflow-diabetes-dataset

## 📹 Demo
[![Watch the demo](https://img.youtube.com/vi/t_nYCwesUq4/0.jpg)](https://www.youtube.com/watch?v=t_nYCwesUq4)

🧠 Streamlit + MLflow Integration Demo
This project demonstrates the seamless integration of MLflow with Streamlit, enabling users to manage the entire machine learning lifecycle — from model selection and training to model registration and prediction — all within a single, user-friendly web interface.

🚀 Features
- Choose from multiple scikit-learn models
- Set custom hyperparameters
- Train models on the Diabetes dataset from sklearn
- Download and register trained models with MLflow
- View model metrics, artifacts, and other details directly from the MLflow UI
- Upload downloaded models and run predictions interactively
- Fully interactive and visually intuitive UI built with Streamlit

Start the MLflow Tracking Server
This will launch the MLflow UI at http://localhost:5000

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

You can access the MLflow UI at:
http://localhost:5000
Use it to inspect experiment runs, metrics, parameters, and artifacts logged during model training.

💡Inspiration
This project serves as a lightweight MLops sandbox, demonstrating how tools like MLflow and Streamlit can come together to create an end-to-end machine learning experience — from experimentation to deployment-ready models.




