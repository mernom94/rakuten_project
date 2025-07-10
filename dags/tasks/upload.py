from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, inspect
from airflow.hooks.base import BaseHook
import pandas as pd
# import boto3
# import os

engine = create_engine("postgresql+psycopg2://rakutenadmin:rakutenadmin@postgres:5432/rakuten_db")

# conn = BaseHook.get_connection('minio_default')

# s3 = boto3.client(
#     's3',
#     endpoint_url=f"http://{conn.host}:{conn.port}",
#     aws_access_key_id=conn.login,
#     aws_secret_access_key=conn.password,
# )
def load_x_to_pg(csv_path, table_name, num_rows):
    df = pd.read_csv(csv_path, skiprows=1, names=[
        "id", "designation", "description", "productid", "imageid"
    ])

    num_rows = min(max(num_rows, 0), len(df))
    df_to_import = df.iloc[:num_rows]
    df_remaining = df.iloc[num_rows:]

    metadata = MetaData(bind=engine)

    Table(
        table_name, metadata,
        Column("id", Integer, primary_key=True),
        Column("designation", String),
        Column("description", String),
        Column("productid", String),
        Column("imageid", String),
    )

    metadata.create_all()

    df_to_import.to_sql(table_name, engine, if_exists="append", index=False)

    df_remaining.to_csv(csv_path, index=False)

def load_y_to_pg(csv_path, table_name, num_rows):
    df = pd.read_csv(csv_path, skiprows=1, names=[
        "id", "prdtypecode"
    ])

    num_rows = min(max(num_rows, 0), len(df))
    df_to_import = df.iloc[:num_rows]
    df_remaining = df.iloc[num_rows:]
    
    metadata = MetaData(bind=engine)

    Table(
        table_name, metadata,
        Column("id", Integer, primary_key=True),
        Column("prdtypecode", Integer),
    )
    
    metadata.create_all()
    
    df_to_import.to_sql(table_name, engine, if_exists="append", index=False)

    df_remaining.to_csv(csv_path, index=False)
    
def drop_pg_tables(table_names: list):
    metadata = MetaData(bind=engine)
    inspector = inspect(engine)

    for table_name in table_names:
        if inspector.has_table(table_name):
            table = Table(table_name, metadata, autoload_with=engine)
            table.drop(engine)
            print(f"Dropped table: {table_name}")
        else:
            print(f"Table '{table_name}' does not exist, skipping.")
    
# def load_x_to_pg(csv_path, table_name, portion):
#     df = pd.read_csv(csv_path, skiprows=1, names=[
#         "id", "designation", "description", "productid", "imageid"
#     ])

#     # only choose portion of the dataset
#     portion = min(max(portion, 0), 1) 
#     num_rows = int(len(df) * portion)
#     df = df.iloc[:num_rows]

#     metadata = MetaData(bind=engine)

#     Table(
#         table_name, metadata,
#         Column("id", Integer, primary_key=True),
#         Column("designation", String),
#         Column("description", String),
#         Column("productid", String),
#         Column("imageid", String),
#     )

#     metadata.create_all()

#     df.to_sql(table_name, engine, if_exists="append", index=False) 
    
    
# def load_y_to_pg(csv_path, table_name, portion):
#     df = pd.read_csv(csv_path, skiprows=1, names=[
#         "id", "prdtypecode"
#     ])

#     # only choose portion of the dataset
#     portion = min(max(portion, 0), 1) 
#     num_rows = int(len(df) * portion)
#     df = df.iloc[:num_rows]
    
#     metadata = MetaData(bind=engine)

#     Table(
#         table_name, metadata,
#         Column("id", Integer, primary_key=True),
#         Column("prdtypecode", Integer),
#     )
    
#     metadata.create_all()
    
#     df.to_sql(table_name, engine, if_exists="append", index=False) 
    
# def load_images(table_name, local_path, bucket_name="rakuten-image"):
    
#     query = f'SELECT imageid, productid FROM "{table_name}"'

#     df = pd.read_sql(query, con=engine)

#     filenames = [
#         f"image_{row.imageid}_product_{row.productid}.jpg"
#         for _, row in df.iterrows()
#     ]
#     upload_count = 0
    
#     for filename in filenames:
#         filepath = os.path.join(local_path, filename)
#         if os.path.exists(filepath):
#             s3.upload_file(filepath, bucket_name, f"{table_name}/{filename}")
#             upload_count += 1
#         else:
#             print(f"File not found: {filepath}")
#     return upload_count
    
