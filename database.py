
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chatbot.db"


def now():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def create_chat(title="New Chat"):
    thread_id = str(uuid.uuid4())
    timestamp = now()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO chats
                (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (thread_id, title, timestamp, timestamp))

    return thread_id


def ensure_chat(thread_id, title="Restored chat", timestamp=None):
    """Create a chat-list entry for a thread already saved by the checkpointer."""
    timestamp = timestamp or now()

    with get_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO chats
                (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (thread_id, title, timestamp, timestamp))


def get_chats():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT thread_id, title, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def update_chat_title(thread_id, title):
    title = " ".join(title.split())[:100]

    if not title:
        return

    with get_connection() as conn:
        conn.execute("""
            UPDATE chats
            SET title = ?, updated_at = ?
            WHERE thread_id = ?
        """, (title, now(), thread_id))


def touch_chat(thread_id):
    with get_connection() as conn:
        conn.execute("""
            UPDATE chats
            SET updated_at = ?
            WHERE thread_id = ?
        """, (now(), thread_id))


def delete_chat_metadata(thread_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM chats
            WHERE thread_id = ?
        """, (thread_id,))

 
init_db()
