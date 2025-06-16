# Rakuten Product Classification - MLOps Pipeline

This project demonstrates a complete MLOps pipeline for the [Rakuten product classification challenge](https://challengedata.ens.fr/participants/challenges/35/), focusing on deployment, versioning, and operational aspects rather than model accuracy.

## Project Overview

- **Task**: Classify products into categories using text and image data
- **Focus**: MLOps infrastructure (MLflow, FastAPI, Docker, model versioning)
- **Models**: Classical ML pipeline with XGBoost, Random Forest, Logistic Regression, and SVM (with plans to integrate deep learning models later)

## Project Structure

```
rakuten_project/
├── config/                  # Configuration files
│   └── airflow.cfg         # Airflow configuration
├── scripts/                 # Setup and data loading scripts
│   ├── 1_download.sh       # Downloads Rakuten dataset
│   ├── 2_unzip_install.sh  # Extracts and installs
│   └── 3_service_load.sh   # Loads data into PostgreSQL + MinIO
├── src/                     # Source code
│   ├── load_minio.py       # MinIO object storage operations
│   ├── load_postgres.py    # PostgreSQL database operations
│   └── (ML pipeline components to be added)
├── .env                    # Environment variables
├── .gitignore
├── docker-compose.yml      # Docker services configuration
├── docker-compose.backup.yaml  # Backup Docker configuration
├── requirements.txt
├── servers.json           # Database server configuration
└── README.md              # This file
```

**Note:** Data is stored in Docker containers (PostgreSQL for metadata, MinIO for images), not in local folders.

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Pockyee/rakuten_project.git
cd rakuten_project
```

### 2. Setup Local Development Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Troubleshooting:**
- **macOS**: If you get "python: command not found", try using `python3`
- **Windows**: If you get "python: command not found", try using `py` or `python3`
- **All platforms**: You can check your Python version with `python --version` or `python3 --version`

### 3. Download Dataset
1. Download the Rakuten product classification dataset
2. Extract and organize files in the existing data structure:
   ```
   data/
   ├── images/          # Place all product images here
   │   ├── image_001.jpg
   │   ├── image_002.jpg
   │   └── ...
   ├── text/            # Place CSV files here
   │   └── product_data.csv
   └── processed/       # Will be populated by preprocessing scripts
       ├── images/
       └── text/
   ```
3. The folder structure is already created in the repo with .gitkeep files

### 4. Setup Database
We use local MongoDB instances for this project for the following reasons:
- Faster queries (no internet latency)
- Works offline during development
- No storage limits or costs
- Each team member can experiment independently
- Simpler initial setup and coordination

**Install and setup local MongoDB:**
1. Install MongoDB locally on your system
2. Start MongoDB service
3. The database connection is configured in the loading script

*Note: We can migrate to a shared MongoDB Atlas instance later if team collaboration requires it.*

### 5. Load Data to Database
```bash
python scripts/load_data_to_db.py
```

### 6. Run the Application
```bash
# Start the API
python -m uvicorn src.api.main:app --reload

# Run MLflow UI
mlflow ui
```

## MLOps Metrics

The following metrics will be tracked and displayed in a Grafana Dashboard (model performance is tracked in MLflow for information only):

**Data Quality Monitoring:**
- Outliers ratio
- Null values ratio  
- Basic statistics
- Unbalance ratio
- Suspicious correlations

**Data Drift Monitoring:**
- Jensen–Shannon Distance
- Population Stability Index (PSI)
- KL Divergence

**Application Metrics:**
- Throughput (Requests/minute)
- Error Rate
- Latency
- CPU usage

## Team Development

Each team member should:
1. Clone this repository
2. Download dataset to local `./data/` subfolders (images in `./data/images/`, text in `./data/text/`)
3. Set up local MongoDB database
4. Run the data loading script
5. Develop and test locally

## Technology Stack

- **Database**: PostgreSQL (containerized)
- **Object Storage**: MinIO (for images)
- **ML Tracking**: MLflow
- **API**: FastAPI
- **Containerization**: Docker
- **Models**: Scikit-learn (XGBoost, Random Forest, Logistic Regression, SVM)
- **Testing**: Pytest
- **Monitoring**: Grafana Dashboard
- **Data Quality/Drift**: Custom monitoring pipeline

## Contributing

1. Create feature branch from main
2. Make changes and test locally
3. Submit pull request
4. Ensure all tests pass