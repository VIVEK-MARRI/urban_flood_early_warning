# airflow/dags/retraining_pipeline.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# Import the logic module from the same dags directory (Option A)
from retrain_logic import run_retraining_pipeline, promote_model_callable 

# Define constants
DEPLOY_MODEL_NAME = "UrbanFloodClassifier"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 20),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "model_retraining_pipeline",
    default_args=default_args,
    description="Automated ML retraining, validation, and promotion via MLflow Model Registry.",
    schedule_interval="0 0 * * 1", # Runs every Monday at midnight (Weekly retraining)
    catchup=False,
) as dag:
    
    # 1. Model Training & Metrics Calculation
    train_task = PythonOperator(
        task_id="train_and_validate",
        python_callable=run_retraining_pipeline,
        op_kwargs={"run_name": f"weekly_retrain_{{{{ ds }}}}"},
        dag=dag,
    )

    # 2. Promotion Logic (XCom pulls the run_id and metrics from train_task)
    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_callable,
        provide_context=True,
        dag=dag,
    )
    
    # 3. Deployment Task (Mocks the successful deployment step)
    deploy_task = BashOperator(
        task_id="deployment_successful",
        bash_command=f"echo 'Deployment initiated for new production model: {DEPLOY_MODEL_NAME}'",
        trigger_rule='one_success',
        dag=dag,
    )
    
    # 4. Skip/Failure Task
    skip_task = EmptyOperator(
        task_id="promotion_skipped",
        dag=dag,
    )

    # Define the pipeline flow: Train -> Promote (conditional branch) -> Deploy OR Skip
    train_task >> promote_task
    promote_task >> [deploy_task, skip_task]