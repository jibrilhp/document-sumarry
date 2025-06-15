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

The application needs GOOGLE_API_KEY to start

You can start the app by running this Makefile

```bash
# production mode
make run
# development mode
make dev-run
```

## API Documentation

You can found API documentation on [here](http://127.0.0.1:8000/docs).

## Conversation Flow

![conversation flow](output_image.png)

The conversation uses iterative refinement to create document summarization. The `refine_summary` will continue to run recursively the model reads all document's from user.

## Chat API explanation

Here is the curl command to call conversation API:

```bash
curl --location '<HOST>>/v1/conversation' \
--header 'Tenant-Id: production-user-1' \
--header 'Api-Key: <API_KEY>' \
--header 'Authorization: Bearer <JWT_TOKEN>>"' \
--form 'message="Kapan masa kepersetaan JPK3 dimulai?"' \
--form 'conversation_uuid="b0dce2a9-a6fa-4267-9959-b0790fe05ba9"' \
--form 'is_stream="false"'
```

`Tenant-Id` can be any string

`project_uuid` is project's id. We can create a project with this API:

```bash
curl --location '<HOST>/v1/projects' \
--header 'Content-Type: application/json' \
--data '{
    "name": "<PROJECT_NAME>"
}'
```

`conversation_uuid` is conversation's id. It distinct a particular conversation from others. It is also crucial for LLM's memory management.

`files` supports JPG/PNG and pdf document

`is_stream` is a toggle for response type. If `is_stream` is `true`, backend will response with streamable response. If it `False`, backend will response with a `text/plain` response.

`Api-Key` is a key that can be generated through this API:

```bash
curl --location 'http://HOST/v1/user/access-token' \
--header 'accept: application/json' \
--header 'Content-Type: application/json' \
--header 'Authorization: <JWT_TOKEN>>' \
--data '{
    "description": "demo key"
}'
```

The conversation API is rate limited. This app use the API-Key to perform rate limiting. The amount of permitted request is configured on `RATE_LIMIT` environment variable.

## Audit Log

This app logs all user activity but `/login` endpoint. The logs would look something like this:

```
2025-06-15 19:52:44,392 - INFO - handler.routes - user=johndoe path=/v1/documents method=GET
```

Especially for conversation API, two additional field is added

```
2025-06-15 19:54:23,087 - INFO - handler.routes - user=johndoe path=/v1/conversation method=POST request_token=1961, response_token=190
```
