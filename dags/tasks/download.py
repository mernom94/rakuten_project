import os, requests

def download_raw_data():
    
    raw_data_dir = '/opt/airflow/raw_data'
    
    cookies = {
        'csrftoken': 'PgtE5ZWyVyULTYn7K54IhQaQ73tcCXqT',
        'sessionid': 'qm863u47vqs8pqor4oariah186ovgpu0',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    }
    
    files = {
        'x_train.csv': 'https://challengedata.ens.fr/participants/challenges/35/download/x-train',
        'y_train.csv': 'https://challengedata.ens.fr/participants/challenges/35/download/y-train',
        'x_test.csv': 'https://challengedata.ens.fr/participants/challenges/35/download/x-test',
        'images.zip': 'https://challengedata.ens.fr/participants/challenges/35/download/supplementary-files',
    }    
    
    for filename, url in files.items():
        dest_path = os.path.join(raw_data_dir, filename)
        print(f"Downloading {filename} ...")
        response = requests.get(url, headers=headers, cookies=cookies)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)