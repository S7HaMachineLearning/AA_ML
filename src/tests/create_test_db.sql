-- Create the 'raw_automations' table
CREATE TABLE raw_automations
(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT UNIQUE,
    data TEXT,
    createdOn TEXT DEFAULT CURRENT_TIMESTAMP,
    updatedOn TEXT,
    deleted INTEGER DEFAULT 0
);

-- Create the 'processed_automations' table
CREATE TABLE processed_automations
(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platforms TEXT,
    conditions TEXT,
    services TEXT,
    createdOn TEXT DEFAULT CURRENT_TIMESTAMP,
    updatedOn TEXT,
    deleted INTEGER DEFAULT 0
);

-- Insert some example data into the 'raw_automations' table
INSERT INTO raw_automations (hash, data)
VALUES ('example_hash_1', '{"platform": "state", "condition": "sun", "service": "light.turn_on"}');

-- Insert some example data into the 'processed_automations' table
INSERT INTO processed_automations (platforms, conditions, services)
VALUES ('["state"]', '["sun"]', '["light.turn_on"]');
