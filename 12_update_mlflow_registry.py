
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

V2_MODEL_PATH = Path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = Path("models/retrained_preprocessor_v2.pkl")

RETRAINING_REPORT_PATH = Path(
    "outputs/retraining/retraining_report.json"
)

COMPARISON_REPORT_PATH = Path(
    "outputs/retraining/model_comparison_report.json"
)

REGISTRY_REPORT_PATH = Path(
    "outputs/retraining/mlflow_registry_update_report.json"
)


# ---------------------------------------------------------
# CONFIGURE MLFLOW
# ---------------------------------------------------------

def configure_mlflow(config):

    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
        or "sqlite:///mlflow.db"
    )

    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


# ---------------------------------------------------------
# LOAD REPORTS
# ---------------------------------------------------------

def load_reports():

    with open(RETRAINING_REPORT_PATH, "r", encoding="utf-8") as f:
        retraining_report = json.load(f)

    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        comparison_report = json.load(f)

    return retraining_report, comparison_report


# ---------------------------------------------------------
# GET CURRENT PRODUCTION VERSION
# ---------------------------------------------------------

def get_production_version(client, model_name):

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    production_versions = [
        version
        for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        return None

    return production_versions[0]


# ---------------------------------------------------------
# REGISTER V2 MODEL
# ---------------------------------------------------------

def register_v2(config, retraining_report, comparison_report):

    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    decision_metric = config["registry"]["decision_metric"]

    v2_metrics = retraining_report["metrics"]
    v1_metrics = comparison_report["model_v1"]["metrics"]

    source_retraining_run_id = retraining_report["mlflow_run_id"]

    print()
    print("CURRENT REGISTRY STATE")
    print("-" * 70)

    production_version = get_production_version(
        client,
        model_name
    )

    if production_version:

        production_run = client.get_run(
            production_version.run_id
        )

        production_metric = production_run.data.metrics.get(
            decision_metric,
            0.0
        )

        print(
            f"Production version : {production_version.version}"
        )

        print(
            f"Production run ID  : {production_version.run_id}"
        )

        print(
            f"Production {decision_metric} : "
            f"{production_metric:.4f}"
        )

    else:

        production_metric = None
        print("Production version : None")

    v2_metric = float(
        v2_metrics[decision_metric]
    )

    print()
    print("V2 REGISTRATION DECISION")
    print("-" * 70)

    print(
        f"V2 {decision_metric} : {v2_metric:.4f}"
    )

    if production_metric is not None:

        print(
            f"Production {decision_metric} : "
            f"{production_metric:.4f}"
        )

    # Quality gate
    minimum_recall = float(
        config["quality_gate"]["minimum_value"]
    )

    recall_value = float(
        v2_metrics["recall"]
    )

    quality_gate_passed = (
        recall_value >= minimum_recall
    )

    print(
        f"Recall quality gate : "
        f"{'PASS' if quality_gate_passed else 'FAIL'}"
    )

    # Compare against Production
    if production_metric is not None:

        improves_production = (
            v2_metric > production_metric
        )

    else:

        improves_production = True

    print(
        f"Improves Production : "
        f"{'YES' if improves_production else 'NO'}"
    )

    # -----------------------------------------------------
    # START DEDICATED MLFLOW RUN
    # -----------------------------------------------------

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="Lab12_Registry_V2"
    ) as run:

        mlflow.log_param(
            "lab",
            "Lab 12 - MLflow Registry Update"
        )

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )

        mlflow.log_param(
            "model_candidate",
            "V2"
        )

        mlflow.log_param(
            "previous_production_version",
            "1"
        )

        mlflow.log_param(
            "source_retraining_run_id",
            source_retraining_run_id
        )

        mlflow.log_param(
            "decision_metric",
            decision_metric
        )

        mlflow.log_param(
            "quality_gate_metric",
            "recall"
        )

        mlflow.log_param(
            "quality_gate_minimum",
            minimum_recall
        )

        mlflow.log_param(
            "training_dataset",
            retraining_report["training_dataset"]
        )

        mlflow.log_param(
            "training_rows",
            retraining_report["training_rows"]
        )

        # Log V2 metrics
        for metric_name, metric_value in v2_metrics.items():

            mlflow.log_metric(
                metric_name,
                float(metric_value)
            )

        mlflow.log_metric(
            "v1_recall",
            float(v1_metrics["recall"])
        )

        mlflow.log_metric(
            "v1_decision_metric",
            float(v1_metrics[decision_metric])
        )

        mlflow.log_metric(
            "v2_vs_v1_decision_metric",
            float(
                v2_metric - float(v1_metrics[decision_metric])
            )
        )

        # -------------------------------------------------
        # REGISTER RANDOM FOREST MODEL
        # -------------------------------------------------

        print()
        print("REGISTERING V2 MODEL")
        print("-" * 70)

        v2_model = joblib.load(V2_MODEL_PATH)

        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="retrained_random_forest_v2",
            registered_model_name=model_name,

            # Trust only the specific sklearn tree type
            # required by this project-created Random Forest.
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("V2 model logged successfully.")

        # -------------------------------------------------
        # LOG PREPROCESSOR AND REPORTS
        # -------------------------------------------------

        mlflow.log_artifact(
            str(V2_PREPROCESSOR_PATH),
            artifact_path="preprocessing_pipeline_v2"
        )

        mlflow.log_artifact(
            str(RETRAINING_REPORT_PATH),
            artifact_path="retraining_reports"
        )

        mlflow.log_artifact(
            str(COMPARISON_REPORT_PATH),
            artifact_path="retraining_reports"
        )

        registry_run_id = run.info.run_id

    # -----------------------------------------------------
    # FIND NEWLY REGISTERED MODEL VERSION
    # -----------------------------------------------------

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.run_id == registry_run_id
    ]

    if not matching_versions:

        raise RuntimeError(
            "Could not find the newly registered V2 model version."
        )

    v2_version = matching_versions[0]

    print(
        f"Registered V2 version: {v2_version.version}"
    )

    # -----------------------------------------------------
    # ADD REGISTRY TAGS
    # -----------------------------------------------------

    client.set_model_version_tag(
        name=model_name,
        version=v2_version.version,
        key="candidate",
        value="V2"
    )

    client.set_model_version_tag(
        name=model_name,
        version=v2_version.version,
        key="source_retraining_run_id",
        value=source_retraining_run_id
    )

    client.set_model_version_tag(
        name=model_name,
        version=v2_version.version,
        key="quality_gate",
        value="PASSED" if quality_gate_passed else "FAILED"
    )

    client.set_model_version_tag(
        name=model_name,
        version=v2_version.version,
        key="comparison_with_production",
        value="IMPROVED" if improves_production else "NOT_IMPROVED"
    )

    # -----------------------------------------------------
    # MOVE V2 TO STAGING
    # DO NOT REPLACE PRODUCTION AUTOMATICALLY
    # -----------------------------------------------------

    client.transition_model_version_stage(
        name=model_name,
        version=v2_version.version,
        stage="Staging",
        archive_existing_versions=False
    )

    # -----------------------------------------------------
    # VERIFY PRODUCTION VERSION
    # -----------------------------------------------------

    versions_after = client.search_model_versions(
        f"name='{model_name}'"
    )

    production_after = [
        version
        for version in versions_after
        if version.current_stage == "Production"
    ]

    production_version_after = (
        production_after[0].version
        if production_after
        else None
    )

    # -----------------------------------------------------
    # GENERATE REGISTRY REPORT
    # -----------------------------------------------------

    report = {
        "stage": "Lab 12 - MLflow Registry Update",
        "model_name": model_name,
        "registered_version": int(v2_version.version),
        "registered_run_id": registry_run_id,
        "source_retraining_run_id": source_retraining_run_id,

        "model_stage": "Staging",

        "production_version_before": (
            int(production_version.version)
            if production_version
            else None
        ),

        "production_version_after": (
            int(production_version_after)
            if production_version_after
            else None
        ),

        "decision_metric": decision_metric,
        "v2_decision_metric": v2_metric,
        "v2_recall": recall_value,

        "recall_quality_gate": (
            "PASSED"
            if quality_gate_passed
            else "FAILED"
        ),

        "improves_production": improves_production,

        "promotion_to_production": False
    }

    REGISTRY_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REGISTRY_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    return report


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("LAB 12.15 - MLFLOW REGISTRY UPDATE")
    print("=" * 70)

    # Check required files
    required_files = [
        V2_MODEL_PATH,
        V2_PREPROCESSOR_PATH,
        RETRAINING_REPORT_PATH,
        COMPARISON_REPORT_PATH
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load configuration
    with open(
        "config.yaml",
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    # Configure MLflow
    configure_mlflow(config)

    # Load reports
    retraining_report, comparison_report = load_reports()

    # Register V2
    report = register_v2(
        config,
        retraining_report,
        comparison_report
    )

    # Display final results
    print()
    print("=" * 70)
    print("REGISTRY UPDATE COMPLETE")
    print("=" * 70)

    print(
        f"Registered model : {report['model_name']}"
    )

    print(
        f"V2 registry version : {report['registered_version']}"
    )

    print(
        f"V2 stage          : {report['model_stage']}"
    )

    print(
        f"Production before : {report['production_version_before']}"
    )

    print(
        f"Production after  : {report['production_version_after']}"
    )

    print(
        f"Recall gate       : {report['recall_quality_gate']}"
    )

    print(
        f"Better than V1    : "
        f"{'YES' if report['improves_production'] else 'NO'}"
    )

    print(
        f"Promoted          : "
        f"{'YES' if report['promotion_to_production'] else 'NO'}"
    )

    print(
        f"Report            : {REGISTRY_REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
