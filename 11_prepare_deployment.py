
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import skops.io as sio
import yaml
from mlflow import MlflowClient


MODEL_NAME = "Telco_Churn_Production_Model"
TRUSTED_MODEL_TYPES = ["sklearn.tree._tree.Tree"]


def load_config():
    project_root = Path(__file__).resolve().parent

    with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return project_root, config


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(
            f"No Production version found for '{model_name}'."
        )

    if len(production_versions) > 1:
        raise RuntimeError(
            f"Multiple Production versions found for '{model_name}'."
        )

    return production_versions[0]



def load_trusted_sklearn_model(model_uri):
    """
    Load an MLflow sklearn model using the model.skops artifact
    and explicitly validate the types that require trust.
    """
    print("[INFO] Downloading MLflow model artifacts...")

    model_dir = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri
        )
    )

    model_file = model_dir / "model.skops"

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Could not find model.skops at {model_file}"
        )

    untrusted_types = sio.get_untrusted_types(
        file=str(model_file)
    )

    unexpected_types = [
        type_name
        for type_name in untrusted_types
        if type_name not in TRUSTED_MODEL_TYPES
    ]

    if unexpected_types:
        raise RuntimeError(
            "Unexpected untrusted model types detected: "
            f"{unexpected_types}. Loading stopped for safety."
        )

    print(f"[INFO] Model types requiring trust: {untrusted_types}")

    model = sio.load(
        str(model_file),
        trusted=TRUSTED_MODEL_TYPES,
    )

    print("[SUCCESS] Production model loaded.")

    return model

def prepare_deployment():
    project_root, config = load_config()

    deployment_dir = project_root / "deployment"

    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)

    deployment_dir.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()

    model_version = get_production_model(client, MODEL_NAME)

    model_name = model_version.name
    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")
    print(f"[INFO] Source     : {model_source}")

    print("[INFO] Loading Production model...")

    model_uri = f"models:/{model_name}/{version}"
    model = load_trusted_sklearn_model(model_uri)

    model_path = deployment_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"[SUCCESS] Model saved to {model_path}")

    print("[INFO] Downloading matching preprocessor...")

    preprocessor_path = client.download_artifacts(
        run_id,
        "preprocessing_pipeline/preprocessor.pkl",
        dst_path=str(deployment_dir),
    )

    downloaded_preprocessor = Path(preprocessor_path)
    final_preprocessor_path = deployment_dir / "preprocessor.pkl"

    if downloaded_preprocessor.resolve() != final_preprocessor_path.resolve():
        shutil.move(
            str(downloaded_preprocessor),
            str(final_preprocessor_path),
        )

    print(f"[SUCCESS] Preprocessor saved to {final_preprocessor_path}")

    print("[INFO] Verifying deployment artifacts...")

    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(final_preprocessor_path)

    if loaded_model is None:
        raise RuntimeError("Model verification failed.")

    if loaded_preprocessor is None:
        raise RuntimeError("Preprocessor verification failed.")

    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": model_version.current_stage,
        "run_id": run_id,
        "model_source": model_source,
        "model_id": model_source.split("/")[-1],
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "source": "MLflow",
    }

    metadata_path = deployment_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Metadata saved to {metadata_path}")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("├── model.pkl")
    print("├── preprocessor.pkl")
    print("└── model_metadata.json")

    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    prepare_deployment()