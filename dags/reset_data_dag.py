from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.download import download_raw_data
# from tasks.utils import unzip_file
from tasks.upload import drop_pg_tables 

with DAG(
    dag_id='reset_data',
    description='reset raw data from Internet',
    tags=['Rakuten'],
    schedule=None,
    default_args={
        'owner': 'airflow',
        "start_date": datetime(2025, 6, 15),
    },
    catchup=False
) as dag:
    
    task_1 = PythonOperator(
        task_id='drop_x_y_train_tables',
        python_callable=drop_pg_tables,
        op_kwargs={'table_names': ['x_train', 'y_train']},
    )
    task_2 = PythonOperator(
        task_id='download_raw_data',
        python_callable=download_raw_data,
    )
    
    # task_2 = PythonOperator(
    #     task_id='unzip_image',
    #     python_callable= unzip_file,
    #     op_kwargs={
    #         'zip_path': "/opt/airflow/raw_data/images.zip",
    #         'extract_to': "/opt/airflow/raw_data/"
    #     }  
    # )
    
    task_1 >> task_2