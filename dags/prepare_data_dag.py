from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.upload import load_xy_to_pg
# from tasks.utils import unzip_file, create_minio_bucket

BATCH_SIZE = 2000

with DAG(
    dag_id='prepare_data',
    description='Prepare data from raw_data directory',
    tags=['Rakuten'],
    schedule="*/15 * * * *",
    default_args={
        'owner': 'airflow',
        "start_date": datetime(2025, 6, 15),
    },
    catchup=False
) as dag:
    
    
    task_1 = PythonOperator(
        task_id='upload_xy_train',
        python_callable=load_xy_to_pg,
        op_kwargs={
            'x_path': "/opt/airflow/raw_data/x_train.csv",
            'y_path': "/opt/airflow/raw_data/y_train.csv",
            'x_table': "x_train",
            'y_table': "y_train",
            'end_row': BATCH_SIZE
        }
    )
    
    # task_1_2 = PythonOperator(
    #     task_id='upload_x_test',
    #     python_callable=load_x_to_pg,
    #     op_kwargs={
    #         'csv_path': "/opt/airflow/raw_data/x_test.csv",
    #         'table_name': "x_test",
    #         'portion': SUBSETPORTION
    #     }
    # )

    


    # task_2 = PythonOperator(
    #     task_id="create_bucket",
    #     python_callable=create_minio_bucket,
    #     op_kwargs={
    #         'bucket_name':"rakuten-image"
    #     }
    # )
    
    # task_3_1 = PythonOperator(
    #     task_id="load_train_image",
    #     python_callable=load_images,
    #     op_kwargs={
    #         'table_name':"x_train",
    #         'local_path':"/opt/airflow/raw_data/images/image_train",
    #     }
    # )
    
    # task_3_2 = PythonOperator(
    #     task_id="load_test_image",
    #     python_callable=load_images,
    #     op_kwargs={
    #         'table_name':"x_test",
    #         'local_path':"/opt/airflow/raw_data/images/image_test",
    #     }
    # )

    # [task_1_1, task_1_2, task_1_3] >> task_3_1
    # [task_1_1, task_1_2, task_1_3] >> task_3_2
    # task_2  >> [task_3_1, task_3_2]