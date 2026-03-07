use rusqlite::{Connection, params};
use anyhow::Result;
use directories::ProjectDirs;
use std::path::PathBuf;

pub struct History {
    conn: Connection,
}

impl History {
    pub fn open() -> Result<Self> {
        let db_path = Self::get_path();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        Ok(Self { conn })
    }

    pub fn add_message(&self, role: &str, content: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO history (role, content) VALUES (?1, ?2)",
            params![role, content],
        )?;
        Ok(())
    }

    pub fn get_recent_messages(&self, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content FROM history ORDER BY timestamp DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;

        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        // Reverse because we want them in chronological order for the model
        messages.reverse();
        Ok(messages)
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM history", [])?;
        Ok(())
    }

    fn get_path() -> PathBuf {
        if let Some(dirs) = ProjectDirs::from("com", "qwen", "qsh") {
            return dirs.data_dir().join("history.db");
        }
        PathBuf::from(".qsh_history.db")
    }
}
