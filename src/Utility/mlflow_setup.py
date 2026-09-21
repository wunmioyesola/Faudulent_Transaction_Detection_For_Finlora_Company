import dagshub
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
load_dotenv(override=True)
import os

# Local Environment MLFLOW Access
#def setup_mlflow():
    #"""Initialize MLflow tracking for the project."""

    #dagshub.init(repo_owner='wunmi.abosede',
                #repo_name='Fraudulent_Transaction_Detection_For_Finlora_Company',
                #mlflow=True)

# Set experiment
#mlflow.set_experiment("Fraudulent_Transaction_Detection_for_Finlora")

# Production Environment MLFLOW Access
def setup_mlflow():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("MLflow Token variable not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com/"
    repo_owner='wunmi.abosede'
    repo_name='Fraudulent_Transaction_Detection_For_Finlora_Company'

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    # Set experiment
mlflow.set_experiment("Fraudulent_Transaction_Detection_for_Finlora")

