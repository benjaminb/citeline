-- Create the database if it does not exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test') THEN
        PERFORM dblink_exec('dbname=postgres', 'CREATE DATABASE test');
    END IF;
END
$$;

-- Connect to the test database
\c test

-- Grant privileges to the user
GRANT ALL PRIVILEGES ON DATABASE test TO bbasseri;

-- Grant privileges on the public schema
GRANT ALL PRIVILEGES ON SCHEMA public TO bbasseri;

-- Create the chunks table if it does not exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'chunks') THEN
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            doi TEXT NOT NULL,
            text TEXT NOT NULL
        );
    END IF;
END
$$;

-- Grant privileges on the chunks table and sequence
GRANT ALL PRIVILEGES ON TABLE chunks TO bbasseri;
GRANT ALL PRIVILEGES ON SEQUENCE chunks_id_seq TO bbasseri;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO bbasseri;