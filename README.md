# Assignment 4 - MLOps Project

## Part 1: GitHub Actions (.github/workflows/mlops.yml)
| Trigger | Job |
|---|---|
| Pull request to main | lint-and-test (flake8 + pytest) |
| Push to main | train (preprocess + train, uploads model artifact) |
| Tag `v*` | deploy-staging (train, build Docker image, push to GHCR, run) |

Commands:
    git push -u origin main                      # train job
    git checkout -b feature/x && git push ...    # open PR -> lint-and-test
    git tag v1.0.0 && git push origin v1.0.0     # deploy-staging

## Part 2: Airflow (airflow/dags/mlops_dag.py)
    python -m venv venv && source venv/bin/activate
    pip install "apache-airflow==2.9.3" -r requirements.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
    export MLOPS_PROJECT_DIR=$(pwd)
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
    export AIRFLOW__CORE__LOAD_EXAMPLES=False
    airflow standalone        # UI at http://localhost:8080 (password printed in terminal)
Trigger `mlops_training_pipeline`. Run 1 deploys (no previous model). Run 2 has the same
accuracy, so `skip_deployment` runs. To demo an improvement, lower `n_estimators` in the first run,
then raise it.
 
Testing the PR pipeline 
Testing the PR pipeline 
