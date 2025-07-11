from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
from docker.types import Mount
import os

PROJECT_ROOT = os.environ.get('AIRFLOW_PROJECT_ROOT')

default_args = {
    'owner': 'airflow',
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'run_drift_detection_every_2h',
    default_args=default_args,
    description='Run data drift detection script every 2 hours',
    schedule='0 */2 * * *',
    start_date=datetime(2025, 7, 9),
    catchup=False,
    tags=['drift', 'monitoring'],
) as dag:

    run_drift = DockerOperator(
        task_id='run_drift_detection',
        image='rakuten-ml:latest',
        api_version='auto',
        command="python scripts/drift_detection.py",  
        docker_url='unix://var/run/docker.sock',
        network_mode='rakuten_project_default',
        mounts=[
            Mount(source=f'{PROJECT_ROOT}/processed_data', target='/app/processed_data', type='bind'),
            Mount(source=f'{PROJECT_ROOT}/models', target='/app/models', type='bind'),
        ],
        environment={
            'PYTHONPATH': '/app',
            'PYTHONUNBUFFERED': '1',
        },
        mount_tmp_dir=False,
    )
