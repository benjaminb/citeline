CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    doi TEXT NOT NULL,
    chunk TEXT NOT NULL,
);