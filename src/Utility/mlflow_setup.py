import dagshub
import mlflow
import mlflow.sklearn

def setup_mlflow():
    """Initialize MLflow tracking for the project."""

dagshub.init(
    repo_owner='wunmi.abosede',
    repo_name='Fraudulent_Transaction_Detection_For_Finlora_Company',
    mlflow=True
)

# Set experiment
mlflow.set_experiment("Fraudulent_Transaction_Detection_for_Finlora")           