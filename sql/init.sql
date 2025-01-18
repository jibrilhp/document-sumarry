CREATE TABLE IF NOT EXISTS documents (
    uuid VARCHAR PRIMARY KEY,
    document_name VARCHAR,
    is_processed BOOLEAN,
    document_type int,
    created_at bigint,
    updated_at bigint
);