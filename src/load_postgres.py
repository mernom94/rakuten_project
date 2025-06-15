from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import pandas as pd

engine = create_engine("postgresql+psycopg2://rakutenadmin:rakutenadmin@localhost:5432/rakuten_db")

def load_X_to_pg(csv_path, table_name):
    df = pd.read_csv(csv_path, skiprows=1, names=[
        "id", "designation", "description", "productid", "imageid"
    ])

    metadata = MetaData()

    Table(
        table_name, metadata,
        Column("id", Integer, primary_key=True),
        Column("designation", String),
        Column("description", String),
        Column("productid", String),
        Column("imageid", String),
    )

    metadata.create_all(engine)

    df.to_sql(table_name, engine, if_exists="append", index=False) 
    
    
def load_y_to_pg(csv_path, table_name):
    df = pd.read_csv(csv_path, skiprows=1, names=[
        "id", "prdtypecode"
    ])
    
    metadata = MetaData()

    Table(
        table_name, metadata,
        Column("id", Integer, primary_key=True),
        Column("prdtypecode", Integer),
    )
    
    metadata.create_all(engine)
    
    df.to_sql(table_name, engine, if_exists="append", index=False) 


if __name__ == "__main__":
    load_X_to_pg("./raw_data/x_train.csv", "X_train")
    load_y_to_pg("./raw_data/y_train.csv", "y_train")
    load_X_to_pg("./raw_data/x_test.csv", "X_test")
    print("All tables uploaded successfully.")