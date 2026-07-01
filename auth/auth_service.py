"""
auth/auth_service.py
---------------------
Login / registration logic backed by SQLite. Passwords are never stored
in plaintext: we salt + hash with PBKDF2-HMAC-SHA256 (stdlib only, no
extra dependency needed).
"""

import hashlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PBKDF2_ITERATIONS, SALT_BYTES
from database.db_utils import get_connection


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return dk.hex()


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Create a new user. Returns (success, message)."""
    username = username.strip()
    if not username or not password:
        return False, "Username र password खाली हुन मिल्दैन।"
    if len(password) < 4:
        return False, "Password कम्तिमा 4 character हुनुपर्छ।"

    conn = get_connection()
    try:
        cur = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone() is not None:
            return False, "यो username पहिले नै लिइसकिएको छ।"

        salt = os.urandom(SALT_BYTES)
        password_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt.hex()),
        )
        conn.commit()
        return True, "Account सफलतापूर्वक बन्यो। अब login गर्नुहोस्।"
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> tuple[bool, dict | None, str]:
    """Verify credentials. Returns (success, user_row_as_dict_or_None, message)."""
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cur.fetchone()
        if row is None:
            return False, None, "User फेला परेन।"

        salt = bytes.fromhex(row["salt"])
        expected_hash = _hash_password(password, salt)
        if expected_hash != row["password_hash"]:
            return False, None, "Password मिलेन।"

        return True, dict(row), "Login सफल भयो।"
    finally:
        conn.close()
