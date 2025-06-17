-- Creates a vector table for BGE Large en (1024 dimensions)
CREATE TABLE IF NOT EXISTS contributions (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(1024) NOT NULL,
    text TEXT NOT NULL,
    pubdate DATE NOT NULL,
    doi VARCHAR(255) NOT NULL
);