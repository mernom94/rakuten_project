from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, inspect
from airflow.hooks.base import BaseHook
from sqlalchemy import text
import pandas as pd
import numpy as np
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
def stratified_sample_df(df, label_col, frac=0.1, random_state=13):
    np.random.seed(random_state)
    sampled_indices = []

    for cls in df[label_col].unique():
        cls_df = df[df[label_col] == cls]
        n_sample = max(1, int(len(cls_df) * frac))
        sampled = cls_df.sample(n=n_sample, replace=False, random_state=random_state)
        sampled_indices.extend(sampled.index)

    return df.loc[sampled_indices], df.drop(sampled_indices)

def load_xy_to_pg(x_path, y_path, x_table, y_table, method='range', start_row=None, end_row=None, frac=0.1):
    
    df_x = pd.read_csv(x_path, skiprows=1, names=[
        "id", "designation", "description", "productid", "imageid"
    ])
    df_y = pd.read_csv(y_path, skiprows=1, names=["id", "prdtypecode"])

    df = pd.merge(df_x, df_y, on="id")

    if method == 'sample':
        df_sampled, df_remaining = stratified_sample_df(df, label_col="prdtypecode", frac=frac)
    else:
        total_rows = len(df)
        start_row = 0 if start_row is None else max(0, start_row)
        end_row = total_rows if end_row is None else min(end_row, total_rows)

        if start_row >= end_row:
            print("Invalid row range.")
            return

        df_sampled = df.iloc[start_row:end_row]
        df_remaining = pd.concat([df.iloc[:start_row], df.iloc[end_row:]], ignore_index=True)

    # Split X and Y
    df_x_sampled = df_sampled[["id", "designation", "description", "productid", "imageid"]]
    df_y_sampled = df_sampled[["id", "prdtypecode"]]

    # ========== Write PostgreSQL ==========
    metadata = MetaData(bind=engine)

    Table(
        x_table, metadata,
        Column("id", Integer, primary_key=True),
        Column("designation", String),
        Column("description", String),
        Column("productid", String),
        Column("imageid", String),
    )

    Table(
        y_table, metadata,
        Column("id", Integer, primary_key=True),
        Column("prdtypecode", Integer),
    )

    metadata.create_all()

    df_x_sampled.to_sql(x_table, engine, if_exists="append", index=False)
    df_y_sampled.to_sql(y_table, engine, if_exists="append", index=False)

    # ========== Write back CSV ==========
    df_remaining[["id", "designation", "description", "productid", "imageid"]].to_csv(x_path, index=False)
    df_remaining[["id", "prdtypecode"]].to_csv(y_path, index=False)

    print(f"[{method.upper()}] Imported {len(df_sampled)} rows. Remaining {len(df_remaining)} saved to CSV.")

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

def read_metrics(**kwargs):
    with engine.connect() as connection:
        n_samples = connection.execute(text("""
            SELECT value FROM metrics WHERE key = 'n_samples'
            ORDER BY timestamp DESC LIMIT 1;
        """)).fetchone()
        n_samples = int(n_samples[0]) if n_samples else 0

        eval_f1 = connection.execute(text("""
            SELECT value FROM model_version_tags WHERE key = 'eval_f1'
            ORDER BY version DESC LIMIT 1;
        """)).fetchone()
        eval_f1 = float(eval_f1[0]) if eval_f1 else 0.0

        x_count = connection.execute(text("""
            SELECT COUNT(*) FROM x_train;
        """)).fetchone()
        x_count = int(x_count[0]) if x_count else 0

        print(f"n_samples={n_samples}, eval_f1={eval_f1}, x_count={x_count}")

        return {
            'n_samples': n_samples,
            'eval_f1': eval_f1,
            'x_count': x_count
        }   
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
    
