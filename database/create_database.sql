CREATE USER bbasseri WITH PASSWORD 'citeline';
CREATE DATABASE citelinedb WITH OWNER bbasseri;
-- GRANT ALL PRIVILEGES ON DATABASE citeline_db TO bbasseri;
\c citelinedb -- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bbasseri;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO bbasseri;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bbasseri;
-- GRANT USAGE ON SCHEMA public TO bbasseri;
-- GRANT CREATE ON SCHEMA public TO bbasseri;
-- ALTER ROLE bbasseri SET search_path TO public;
CREATE EXTENSION vector;
-- Allows preloading data into shared buffers for faster queries
CREATE EXTENSION pg_prewarm;
CREATE TABLE library (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    doi VARCHAR(255) NOT NULL,
    pubdate DATE,
    text TEXT NOT NULL
);