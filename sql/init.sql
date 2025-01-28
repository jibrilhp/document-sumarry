CREATE TABLE IF NOT EXISTS document_types (
    id SERIAL PRIMARY key,
    "name" VARCHAR
)

CREATE TABLE IF NOT EXISTS projects (
    uuid VARCHAR PRIMARY key,
    name VARCHAR
)

CREATE TABLE IF NOT EXISTS documents (
    uuid VARCHAR PRIMARY KEY,
    document_name VARCHAR,
    is_processed BOOLEAN,
    document_type_id int REFERENCES document_types(id),
    created_at bigint,
    updated_at bigint,
    projects_uuid VARCHAR REFERENCES projects(uuid),
    tenant_id varchar
);
