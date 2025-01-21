# Document Summarisation Tools

## Introduction

This project uses `pgvector` to store information. The `pgvector` can be installed by running the docker compose

```yaml
docker compose up -d
```

After the pgvector is running, login to postgres container and copy paste the `sql/init.sql` to create `documents` table

This project use `uv` to manage dependency. To install `uv` to your machine, please refer to [uv's installation manual](https://docs.astral.sh/uv/getting-started/installation/).

You can download the project's depndency with this command

```bash
uv sync
```

## How to start the app

You can start the app by running this Makefile

```bash
make run
```

## API Documentation

You can found the API documentation [here](docs/Summarisation App.postman_collection.json). For the summarization endpoint, you can quickly use this curl to try the endpoint.

```bash
curl --location 'http://localhost:5000/v1/summarize' --form 'document=@"<FILE_PATH>"'
```
