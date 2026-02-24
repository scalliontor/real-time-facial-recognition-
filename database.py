"""
database.py — SQLite database module for facial recognition system.

Stores user profiles and their 512-D face embeddings.
Embeddings are stored as BLOBs (numpy arrays serialized via tobytes()).
"""

import sqlite3
import numpy as np
import os
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "faces.db")


def init_db():
    """Initialize the database and create tables if they don't exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def add_user(user_id: str, embedding: np.ndarray):
    """
    Add a new user with their master embedding to the database.

    Args:
        user_id: Unique identifier for the user.
        embedding: 512-D normalized numpy array (float32).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    created_at = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO users (id, created_at) VALUES (?, ?)",
        (user_id, created_at),
    )

    # Store embedding as raw bytes (512 floats × 4 bytes = 2048 bytes)
    embedding_bytes = embedding.astype(np.float32).tobytes()
    cursor.execute(
        "INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
        (user_id, embedding_bytes),
    )

    conn.commit()
    conn.close()
    print(f"[DB] Added user ID: {user_id}")


def get_all_embeddings():
    """
    Retrieve all user embeddings from the database.

    Returns:
        List of tuples: (user_id, embedding_ndarray)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT u.id, e.embedding
        FROM users u
        JOIN embeddings e ON u.id = e.user_id
    """)

    results = []
    for row in cursor.fetchall():
        user_id, embedding_bytes = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        results.append((user_id, embedding))

    conn.close()
    return results


def get_all_users():
    """
    Retrieve all users (without embeddings) from the database.

    Returns:
        List of tuples: (user_id, created_at)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, created_at FROM users ORDER BY created_at DESC")
    results = cursor.fetchall()

    conn.close()
    return results


def get_user(user_id: str):
    """
    Retrieve a specific user by ID.

    Returns:
        Tuple (user_id, created_at) or None if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, created_at FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()

    conn.close()
    return result


def delete_user(user_id: str):
    """Delete a user and their embedding from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Delete embedding first (foreign key)
    cursor.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()
    print(f"[DB] Deleted user ID: {user_id}")



