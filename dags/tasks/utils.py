import zipfile
import os

def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
        
        
from airflow.hooks.base import BaseHook
from airflow.decorators import task
import boto3
from botocore.exceptions import ClientError

def create_minio_bucket(bucket_name):
    conn = BaseHook.get_connection("minio_default")

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{conn.host}:{conn.port}",
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
    )

    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"Bucket '{bucket_name}' already exists.")
        else:
            raise e