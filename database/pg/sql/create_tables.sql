CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(1024) NOT NULL,
    text TEXT NOT NULL,
    doi TEXT NOT NULL,
    pubdate DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_pubdate ON chunks(pubdate);
GRANT ALL ON TABLE chunks TO bbasseri;
GRANT USAGE,
    SELECT ON SEQUENCE chunks_id_seq TO bbasseri;

CREATE TABLE IF NOT EXISTS contributions (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(1024) NOT NULL,
    text TEXT NOT NULL,
    doi TEXT NOT NULL,
    pubdate DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contributions_pubdate ON contributions(pubdate);
GRANT ALL ON TABLE contributions TO bbasseri;
GRANT USAGE,
    SELECT ON SEQUENCE contributions_id_seq TO bbasseri;