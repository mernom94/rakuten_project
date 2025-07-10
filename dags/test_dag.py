from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text
from datetime import datetime

def read_metrics():

    engine = create_engine("postgresql+psycopg2://rakutenadmin:rakutenadmin@postgres:5432/rakuten_db")

    with engine.connect() as connection:

        n_samples_result = connection.execute(text("""
            SELECT value
            FROM metrics
            WHERE key = 'n_samples'
            ORDER BY timestamp DESC
            LIMIT 1;
        """)).fetchone()
        n_samples = n_samples_result[0] if n_samples_result else None

        eval_f1_result = connection.execute(text("""
            SELECT value
            FROM model_version_tags
            WHERE key = 'eval_f1'
            ORDER BY version DESC
            LIMIT 1;
        """)).fetchone()
        eval_f1 = eval_f1_result[0] if eval_f1_result else None


        x_count_result = connection.execute(text("""
            SELECT COUNT(*) FROM x_train;
        """)).fetchone()
        x_count = x_count_result[0] if x_count_result else None

        print(f"Latest n_samples: {n_samples}")
        print(f"Latest eval_f1: {eval_f1}")
        print(f"Total entries in table x: {x_count}")

        return {
            'n_samples': n_samples,
            'eval_f1': eval_f1,
            'x_count': x_count
        }

with DAG(
    dag_id='test',
    description='readdatafrompq',
    tags=['Rakuten'],
    schedule=None,
    default_args={
        'owner': 'airflow',
        "start_date": datetime(2025, 6, 15),
    },
    catchup=False
) as dag:

    read_metrics_task = PythonOperator(
        task_id='read_metrics_from_postgres',
        python_callable=read_metrics,
    )