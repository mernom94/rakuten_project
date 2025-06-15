## Steps to run

1. Clone the repo and go to the project root:

```bash
git clone https://github.com/Pockyee/rakuten_project.git
cd rakuten_project
```

2. Make scripts executable:

```bash
chmod +x ./scripts/*
```

3. Run scripts in order:

```bash
./scripts/1_download.sh
./scripts/2_unzip_install.sh
./scripts/3_service_load.sh
```

## Access services

pgAdmin: http://localhost:8081
MinIO: http://localhost:9001

If accessing from another machine, replace localhost with your server’s IP address.

Credentials are in the docker-compose.yml file.
