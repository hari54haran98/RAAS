"""
DAY 22: MLflow Experiment Tracking for RAAS
Tracks all experiments, parameters, and metrics
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


class RAASExperimentTracker:
    """Tracks RAAS experiments with MLflow."""

    def __init__(self, experiment_name="RAAS_Banking_QA"):
        print("=" * 60)
        print("DAY 22: MLFLOW EXPERIMENT TRACKER")
        print("=" * 60)

        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_id = None

        # Create directories
        Path("mlruns").mkdir(exist_ok=True)
        Path("data/experiments").mkdir(exist_ok=True)

        print(f"✓ Experiment: {experiment_name}")
        print(f"✓ MLflow runs dir: mlruns/")
        print("=" * 60)

    def start_run(self, run_name=None):
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        print(f"\n🚀 Started run: {run_name}")
        print(f"   Run ID: {self.run_id}")
        return self.run

    def log_params(self, params):
        """Log parameters for the current run."""
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"✓ Logged {len(params)} parameters")

    def log_metrics(self, metrics):
        """Log metrics for the current run."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        print(f"✓ Logged {len(metrics)} metrics")

    def log_artifact(self, file_path):
        """Log an artifact file."""
        mlflow.log_artifact(file_path)
        print(f"✓ Logged artifact: {file_path}")

    def log_dict(self, dictionary, file_name):
        """Log a dictionary as JSON artifact."""
        mlflow.log_dict(dictionary, file_name)
        print(f"✓ Logged dict as {file_name}")

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        print(f"\n✓ Run ended: {self.run_id}")

    def log_retrieval_experiment(self, experiment_data):
        """
        Log a complete retrieval experiment.

        Args:
            experiment_data: Dictionary with:
                - params: retrieval parameters
                - metrics: performance metrics
                - results: test results
                - config: configuration used
        """
        with mlflow.start_run(run_name="retrieval_experiment"):
            # Log parameters
            if 'params' in experiment_data:
                self.log_params(experiment_data['params'])

            # Log metrics
            if 'metrics' in experiment_data:
                self.log_metrics(experiment_data['metrics'])

            # Log results as artifact
            if 'results' in experiment_data:
                results_file = "data/experiments/results.json"
                with open(results_file, 'w') as f:
                    json.dump(experiment_data['results'], f, indent=2)
                self.log_artifact(results_file)

            # Log config
            if 'config' in experiment_data:
                self.log_dict(experiment_data['config'], "config.json")

    def get_best_runs(self, metric="test_accuracy", n=5):
        """Get the best n runs by metric."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs([experiment.experiment_id])

        if metric in runs.columns:
            best_runs = runs.nlargest(n, metric)[['run_id', metric, 'status']]
            return best_runs
        else:
            print(f"⚠️ Metric '{metric}' not found")
            return runs.head(n)

    def compare_experiments(self, run_ids, metrics):
        """Compare multiple runs by metrics."""
        comparison = {}
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            comparison[run_id] = {
                metric: run.data.metrics.get(metric) for metric in metrics
            }
        return pd.DataFrame(comparison).T


# Quick test
if __name__ == "__main__":
    tracker = RAASExperimentTracker()

    # Start a test run
    tracker.start_run("test_retrieval")

    # Log some test parameters
    tracker.log_params({
        "chunk_size": 310,
        "retrieval_k": 20,
        "rerank_top_k": 3,
        "alpha": 0.2,
        "model": "all-MiniLM-L6-v2"
    })

    # Log test metrics - FIXED: removed @ symbols
    tracker.log_metrics({
        "precision_3": 0.92,  # was precision@3
        "recall_5": 0.88,  # was recall@5
        "hallucination_rate": 0.05,
        "avg_latency_ms": 1850
    })

    # Log a test result
    test_results = {
        "penalty_query": {
            "found": True,
            "confidence": 0.95,
            "source": "sbi_home_loan_terms p2"
        },
        "document_query": {
            "found": True,
            "confidence": 0.80
        }
    }

    results_file = "data/experiments/test_results.json"
    Path("data/experiments").mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    tracker.log_artifact(results_file)

    tracker.end_run()

    print("\n📊 Recent runs:")
    print(tracker.get_best_runs(metric="precision_3"))