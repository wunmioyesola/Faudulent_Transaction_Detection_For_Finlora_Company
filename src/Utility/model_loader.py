import sys
import os

# This tells Python where your main project root foilder is located
sys.path.append(os.path.dirname(__file__))

from mlflow_setup import setup_mlflow
import mlflow.sklearn

def load_registered_model():
    """ load registered model from mlflow """

    setup_mlflow()

    model_name = "Fraud_Detection_XGBoost_Pipeline"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    return model


if __name__ == "__main__":
    loaded_model = load_registered_model()
    print("Model loaded successfully:", loaded_model)
