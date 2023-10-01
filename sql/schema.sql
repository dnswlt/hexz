-- Table definitions for the PostgreSQL database.
--
-- To create the schema, run sth like:
-- psql -f schema.sql -h localhost hexz hexz 

DROP TABLE IF EXISTS games;
CREATE TABLE games (
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    game_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    -- Identifies the player hosting the game.
    host_id TEXT,
    host_name TEXT,

    PRIMARY KEY (game_id)
);

DROP TABLE IF EXISTS game_history;
DROP SEQUENCE IF EXISTS game_history_seq;
CREATE SEQUENCE game_history_seq;
CREATE TABLE game_history (
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    seqnum INTEGER NOT NULL DEFAULT nextval('game_history_seq'),
    game_id TEXT NOT NULL,
    game_state bytea, -- A serialized GameState proto. NULL for 'undo' and 'redo' entries.
    entry_type TEXT NOT NULL,  -- One of 'reset', 'join', 'move', 'undo', 'redo'.

    PRIMARY KEY (game_id, seqnum)
);

DROP TABLE IF EXISTS wasm_stats;
CREATE TABLE wasm_stats (
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    game_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    move_num INTEGER,
    -- MCTS stats.
    tree_size INTEGER,
    max_depth INTEGER,
    iterations INTEGER,
    elapsed_seconds DOUBLE PRECISION,
    -- Memory stats.
    total_alloc_mib DOUBLE PRECISION,
    heap_alloc_mib DOUBLE PRECISION,
    -- User info.
    user_agent TEXT,
    lang TEXT,
    resolution_width INTEGER,
    resolution_height INTEGER,
    viewport_width INTEGER,
    viewport_height INTEGER,
    browser_window_width INTEGER,
    browser_window_height INTEGER,
    hardware_concurrency INTEGER
);
CREATE INDEX wasm_stats_game_id_idx ON wasm_stats (game_id);
