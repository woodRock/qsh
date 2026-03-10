use anyhow::Result;
use directories::ProjectDirs;
use rusqlite::{Connection, params};
use std::path::PathBuf;

pub struct History {
    conn: Connection,
}

impl History {
    pub fn open() -> Result<Self> {
        Self::open_at(Self::get_path())
    }

    pub fn open_at(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                outcome TEXT DEFAULT 'none',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Migration: Add outcome column if it doesn't exist
        {
            let mut stmt = conn.prepare("PRAGMA table_info(history)")?;
            let columns: Vec<String> = stmt
                .query_map([], |row| {
                    let name: String = row.get(1)?;
                    Ok(name.to_lowercase())
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            if !columns.contains(&"outcome".to_string()) {
                conn.execute("ALTER TABLE history ADD COLUMN outcome TEXT DEFAULT 'none'", [])?;
            }
        }

        Ok(Self { conn })
    }

    pub fn add_message(&self, role: &str, content: &str, outcome: Option<&str>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO history (role, content, outcome) VALUES (?1, ?2, ?3)",
            params![role, content, outcome.unwrap_or("none")],
        )?;
        Ok(())
    }

    pub fn update_last_outcome(&self, outcome: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE history SET outcome = ?1 WHERE id = (SELECT MAX(id) FROM history WHERE role = 'assistant')",
            params![outcome],
        )?;
        Ok(())
    }

    pub fn get_recent_messages(&self, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT role, content FROM history ORDER BY id DESC LIMIT ?1")?;
        let rows = stmt.query_map(params![limit], |row| Ok((row.get(0)?, row.get(1)?)))?;

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

    pub fn get_path() -> PathBuf {
        if let Some(dirs) = ProjectDirs::from("com", "qwen", "qsh") {
            return dirs.data_dir().join("history.db");
        }
        PathBuf::from(".qsh_history.db")
    }
}
