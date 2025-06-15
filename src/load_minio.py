from minio import Minio
import os
from tqdm import tqdm

minio_client = Minio(
    "localhost:9000",
    access_key="rakutenadmin",
    secret_key="rakutenadmin",
    secure=False
)

bucket_name = "rakuten-images"

if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    
def collect_all_files(base_dir):
    file_list = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            local_path = os.path.join(root, file)
            object_name = os.path.relpath(local_path, base_dir).replace("\\", "/")
            file_list.append((local_path, object_name))
    return file_list

def upload_with_progress(file_list, bucket_name):
    for local_path, object_name in tqdm(file_list, desc="Load Progress", unit="file"):
        minio_client.fput_object(bucket_name, object_name, local_path)


if __name__ == "__main__":
    files_to_upload = collect_all_files("./raw_data/images")
    upload_with_progress(files_to_upload, bucket_name)
    print("All images uploaded successfully.")
