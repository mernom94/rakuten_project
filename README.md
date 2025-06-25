# Rakuten Product Classification - MLOps Pipeline

This project demonstrates a complete MLOps pipeline for the [Rakuten product classification challenge](https://challengedata.ens.fr/participants/challenges/35/), focusing on deployment, versioning, and operational aspects rather than model accuracy.

## Project Overview

- **Task**: Classify products into categories using text and image data
- **Focus**: MLOps infrastructure (MLflow, FastAPI, Docker, model versioning)
- **Models**: Classical ML pipeline with XGBoost, Random Forest, Logistic Regression, and SVM (with plans to integrate deep learning models later)
- **Current Performance**: SVM achieves 73.4% F1 score on French text classification

## Project Structure

```
rakuten_project/
├── config/                  # Configuration files
│   └── airflow.cfg         # Airflow configuration
├── dags/                    # Airflow DAGs for workflow orchestration
│   ├── ml_pipeline_dag.py  # PythonOperator ML pipeline
│   ├── ml_pipeline_docker_dag.py  # DockerOperator ML pipeline
│   ├── prepare_data_dag.py # Data preparation workflow
│   └── tasks/              # Airflow task modules
│       ├── ml_tasks.py     # ML pipeline tasks
│       ├── upload.py       # Data upload functions
│       └── utils.py        # Utility functions
├── scripts/                 # Setup and ML scripts
│   ├── 1_install_docker.sh # Install Docker and dependencies
│   ├── 2_download.sh       # Downloads Rakuten dataset
│   ├── 3_run_docker.sh     # Start Docker services
│   ├── preprocessing.py    # Text preprocessing and feature extraction
│   └── training.py         # Model training with GridSearchCV
├── processed_data/          # Generated ML features and metadata
├── models/                  # Trained models and encoders
├── raw_data/               # Original dataset storage
├── src/                     # Source code (legacy)
│   ├── load_minio.py       # MinIO object storage operations
│   ├── load_postgres.py    # PostgreSQL database operations
│   └── (API components to be added)
├── Dockerfile              # ML container definition
├── requirements_ml.txt     # ML-specific dependencies
├── .env                    # Environment variables
├── .gitignore
├── docker-compose.yml      # Docker services configuration
├── requirements.txt        # Main project dependencies
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

### 2. Setup Infrastructure, Environment
Follow the automated setup process:

1. **Make scripts executable:**
   ```bash
   chmod +x ./scripts/*
   ```

2. **Run setup scripts in order:**
   ```bash
   ./scripts/1_install_docker.sh      # Install Docker and unzip utility
   ./scripts/2_download.sh            # Download datasets and unzip images package
   ./scripts/3_run_docker.sh          # Start Docker containers
   ```
   *Some machines may not be able to start all Docker containers simultaneously due to performance limitations. Please rerun 3_run_docker.sh or directly run `docker compose up -d` in the project root directory.*

3. **Access services:**

- **Airflow** (Workflow GUI): [http://localhost:8080](http://localhost:8080)  
  - **Username:** `airflow`  
  - **Password:** `airflow`

- **pgAdmin** (database GUI): [http://localhost:8081](http://localhost:8081)  
  - **Email:** `rakuten@admin.com`  
  - **Password:** `rakutenadmin`

- **MinIO** (object storage GUI): [http://localhost:9001](http://localhost:9001)  
  - **Username:** `rakutenadmin`  
  - **Password:** `rakutenadmin`
   - *If accessing from another machine, replace localhost with your server's IP address*

**Note:** The scripts handle system packages, Docker installation, and data loading.

### 3. Load Data

Enter the Airflow UI and run the prepare_data DAG. After it completes successfully, the XCom under load_test_image should return the value 2762, and the XCom under load_train_image should return the value 16983. By default, 20% of the data is loaded into the database.

### 4. Build ML Container

```bash
# Build the containerized ML environment
docker build -t rakuten-ml:latest .
```

### 5. Run ML Pipeline

#### Option A: Direct Container Execution
```bash
# Preprocessing: Extract features from French text data
docker run --rm --network rakuten_project_default \
  -v $(pwd):/app -w /app \
  rakuten-ml:latest python scripts/preprocessing.py

# Training: Train models with GridSearchCV
docker run --rm --network rakuten_project_default \
  -v $(pwd):/app -w /app \
  rakuten-ml:latest python scripts/training.py
```

#### Option B: Airflow DAG
1. Access Airflow UI: [http://localhost:8080](http://localhost:8080)
2. Trigger DAG: `ml_pipeline_docker`
3. Monitor execution in the UI

### 6. Run the Application (Future)
```bash
# Start the API
python -m uvicorn src.api.main:app --reload

# Run MLflow UI
mlflow ui
```

## ML Pipeline Components

### Text Preprocessing (`scripts/preprocessing.py`)
- **Input**: Raw French product descriptions from PostgreSQL
- **Processing**: 
  - Text cleaning and French stopword removal
  - Multiple text versions (raw, classical ML, BERT-ready)
  - TF-IDF feature extraction (1000 features)
- **Output**: Processed features, targets, and vectorizer saved to `processed_data/`

### Model Training (`scripts/training.py`)
- **Input**: Preprocessed features from previous step
- **Algorithms**: Random Forest, Logistic Regression, SVM, XGBoost
- **Optimization**: GridSearchCV with 3-fold cross-validation
- **Evaluation**: Weighted F1 score, accuracy, classification report
- **Output**: Best model and metadata saved to `models/`

### Current ML Performance
- **Dataset**: 16,983 French product descriptions across 27 categories
- **Best Algorithm**: SVM with linear kernel
- **Test F1 Score**: 73.4%
- **Test Accuracy**: 73.1%
- **Cross-validation Score**: 71.6%

## Docker Architecture

### ML Container (`rakuten-ml:latest`)
- **Base**: Python 3.9 slim
- **Dependencies**: scikit-learn, nltk, xgboost, pandas, psycopg2, etc.
- **Purpose**: Isolated environment for ML workloads with all required dependencies
- **Usage**: Runs both preprocessing and training scripts

### Infrastructure Containers
- **Airflow**: Workflow orchestration and scheduling
- **PostgreSQL**: Structured data storage (text features, metadata)
- **MinIO**: Object storage for images and large files
- **Redis**: Airflow message broker and caching

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

**Model Performance Tracking:**
- F1 Score, Accuracy, Precision, Recall
- Hyperparameter configurations
- Training time and resource usage
- Model versions and deployment history

## Technology Stack

- **Orchestration**: Apache Airflow
- **Database**: PostgreSQL (containerized)
- **Object Storage**: MinIO (for images)
- **ML Pipeline**: Docker containers with scikit-learn stack
- **ML Tracking**: MLflow (in development)
- **API**: FastAPI (in development)
- **Containerization**: Docker and Docker Compose
- **Models**: Scikit-learn (XGBoost, Random Forest, Logistic Regression, SVM)
- **Text Processing**: NLTK, TF-IDF vectorization
- **Testing**: Pytest
- **Monitoring**: Grafana Dashboard (planned)
- **Data Quality/Drift**: Custom monitoring pipeline (planned)

## Development Workflow

### For ML Development
1. Create feature branch from main
2. Develop ML scripts in `scripts/` directory
3. Test using Docker container: `docker run --rm --network rakuten_project_default -v $(pwd):/app -w /app rakuten-ml:latest python scripts/your_script.py`
4. Create or update Airflow DAGs in `dags/`
5. Test DAG execution in Airflow UI
6. Submit pull request with comprehensive testing

### For Infrastructure Development
1. Modify Docker configurations or add new services
2. Test with `docker-compose up -d`
3. Verify service interactions
4. Update documentation and README
5. Submit pull request

## Team Development

Each team member should:
1. Clone this repository
2. Set up local development environment (virtual environment + requirements)
3. Run setup scripts to initialize Docker infrastructure
4. Build ML container for ML development work
5. Develop and test locally using containerized approach

## Troubleshooting

### Common Issues

**Container build fails:**
```bash
# Clear Docker cache and rebuild
docker system prune -f
docker build --no-cache -t rakuten-ml:latest .
```

**ML scripts can't connect to database:**
- Ensure containers are on same network: `rakuten_project_default`
- Check PostgreSQL is running: `docker ps | grep postgres`
- Verify database contains data by running prepare_data DAG

**Airflow DAG not appearing:**
- Check DAG syntax: `python dags/ml_pipeline_docker_dag.py`
- Refresh Airflow UI
- Check Airflow scheduler logs

**Package installation issues:**
- Use Docker approach instead of installing packages on host
- Check `requirements_ml.txt` for ML-specific dependencies
- Ensure NLTK data downloads correctly in container

## Contributing

1. Create feature branch from main
2. Make changes and test locally using Docker containers
3. Ensure all ML scripts work in containerized environment
4. Update documentation as needed
5. Submit pull request with detailed description
6. Ensure all tests pass and DAGs execute successfully

## Future Enhancements

- **MLflow Integration**: Complete model tracking and registry
- **FastAPI Service**: Model serving endpoints with MLflow integration
- **Streamlit Dashboard**: ML results visualization and monitoring
- **Model Versioning**: Automated model deployment pipeline
- **Performance Monitoring**: Real-time model drift detection
- **Deep Learning**: Integration of BERT and image classification models