CREATE TABLE public.users (
	id serial4 NOT NULL,
	email varchar NULL,
	username varchar NULL,
	hashed_password varchar NULL,
	"role" public.userrole NULL,
	is_active bool NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	status int4 DEFAULT 1 NULL,
	CONSTRAINT users_pkey PRIMARY KEY (id)
);
CREATE UNIQUE INDEX ix_users_email ON public.users USING btree (email);
CREATE INDEX ix_users_id ON public.users USING btree (id);
CREATE UNIQUE INDEX ix_users_username ON public.users USING btree (username);

CREATE TABLE IF NOT EXISTS document_types (
    id SERIAL PRIMARY key,
    "name" VARCHAR
);

CREATE TABLE IF NOT EXISTS projects (
    uuid VARCHAR PRIMARY key,
    name VARCHAR
);

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

INSERT INTO document_types (id, "name") VALUES(1, 'PDF');
INSERT INTO document_types(id, "name") VALUES(2, 'Image');

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY key,
    api_key VARCHAR NOT NULL UNIQUE,
    description VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id bigint NOT NULL REFERENCES users(id)
);

-- 26072025: `documents` table no longer needs to assure consistency with `projects` table as it already validated on downstream application
ALTER TABLE public.documents DROP CONSTRAINT documents_projects_uuid_fkey;

-- 26072025: Add column_metadata to datasets table
ALTER TABLE public.datasets ADD column_metadata text NULL;
