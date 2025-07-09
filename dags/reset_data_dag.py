from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.download import download_raw_data
# from tasks.utils import unzip_file
from tasks.upload import load_xy_to_pg, drop_pg_tables

TEST_SET_FRACTION = 0.05

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
        task_id='drop_former_pg_tables',
        python_callable=drop_pg_tables,
        op_kwargs={'table_names': ['x_train', 'y_train', 'x_test', 'y_test']},
    )
    task_2 = PythonOperator(
        task_id='download_raw_data',
        python_callable=download_raw_data,
    )
    
    task_3 = PythonOperator(
        task_id='split_xy_test',
        python_callable=load_xy_to_pg,
        op_kwargs={
            'x_path': "/opt/airflow/raw_data/x_train.csv",
            'y_path': "/opt/airflow/raw_data/y_train.csv",
            'x_table': "x_test",
            'y_table': "y_test",
            'method': "sample",
            'frac': TEST_SET_FRACTION
        }
    )

    # task_2 = PythonOperator(
    #     task_id='unzip_image',
    #     python_callable= unzip_file,
    #     op_kwargs={
    #         'zip_path': "/opt/airflow/raw_data/images.zip",
    #         'extract_to': "/opt/airflow/raw_data/"
    #     }  
    # )
    
    task_1 >> task_2 >> task_3