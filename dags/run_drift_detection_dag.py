from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'run_drift_detection_every_2h',
    default_args=default_args,
    description='Run data drift detection script every 2 hours',
    schedule_interval='0 */2 * * *',  # every 2 hours at minute 0
    start_date=datetime(2025, 7, 9, 0, 0),
    catchup=False,
    tags=['drift', 'monitoring'],
) as dag:

    run_drift_detection = BashOperator(
        task_id='run_drift_detection_script',
        bash_command='python /opt/airflow/dags/drift_detection/drift_detection.py',
    )
