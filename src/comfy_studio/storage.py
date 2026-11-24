from __future__ import annotations

from contextlib import contextmanager
import re
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .permissions import set_private_permissions


_CONVERSATION_ID = re.compile(r"^[0-9a-f]{32}$")
_SOURCE_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")


class StorageError(RuntimeError):
    """Persistent app state is missing or invalid."""


class StorageConflictError(StorageError):
    """A persistent operation conflicts with an existing relationship."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        set_private_permissions(self.path.parent, directory=True)
        try:
            with self.path.open("xb"):
                pass
        except FileExistsError:
            pass
        set_private_permissions(self.path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    context_saved_at TEXT,
                    context_tokens INTEGER,
                    context_bytes INTEGER,
                    context_message_count INTEGER,
                    image_prompt TEXT NOT NULL DEFAULT '',
                    aspect_ratio TEXT NOT NULL DEFAULT '1:1 (Square)',
                    image_mode TEXT NOT NULL DEFAULT 'text-to-image'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    display_content TEXT,
                    reasoning_content TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_id_id
                    ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    prompt TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    aspect_ratio TEXT NOT NULL DEFAULT '1:1 (Square)',
                    megapixels REAL NOT NULL DEFAULT 4.0,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sampler_name TEXT NOT NULL DEFAULT 'euler',
                    scheduler TEXT NOT NULL DEFAULT 'simple',
                    status TEXT NOT NULL,
                    image_mode TEXT NOT NULL DEFAULT 'text-to-image',
                    prompt_id TEXT,
                    filename TEXT,
                    subfolder TEXT,
                    image_type TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS generations_conversation_id_created_at
                    ON generations(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS source_images (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    subfolder TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS source_images_conversation_id_position
                    ON source_images(conversation_id, position);
                CREATE TABLE IF NOT EXISTS generation_sources (
                    generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    source_image_id TEXT NOT NULL REFERENCES source_images(id) ON DELETE RESTRICT,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (generation_id, position),
                    UNIQUE (generation_id, source_image_id)
                );
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
            if "context_message_count" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN context_message_count INTEGER")
            if "image_prompt" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN image_prompt TEXT NOT NULL DEFAULT ''")
            if "aspect_ratio" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN aspect_ratio TEXT NOT NULL DEFAULT '1:1 (Square)'")
            if "image_mode" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN image_mode TEXT NOT NULL DEFAULT 'text-to-image'")
            message_columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
            if "display_content" not in message_columns:
                connection.execute("ALTER TABLE messages ADD COLUMN display_content TEXT")
            if "reasoning_content" not in message_columns:
                connection.execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT NOT NULL DEFAULT ''")
            generation_columns = {row["name"] for row in connection.execute("PRAGMA table_info(generations)")}
            for name, declaration in (
                ("aspect_ratio", "TEXT NOT NULL DEFAULT '1:1 (Square)'"),
                ("megapixels", "REAL NOT NULL DEFAULT 4.0"),
                ("sampler_name", "TEXT NOT NULL DEFAULT 'euler'"),
                ("scheduler", "TEXT NOT NULL DEFAULT 'simple'"),
                ("image_mode", "TEXT NOT NULL DEFAULT 'text-to-image'"),
            ):
                if name not in generation_columns:
                    connection.execute(f"ALTER TABLE generations ADD COLUMN {name} {declaration}")
        set_private_permissions(self.path)

    @staticmethod
    def validate_id(conversation_id: str) -> str:
        if not isinstance(conversation_id, str) or not _CONVERSATION_ID.fullmatch(conversation_id):
            raise StorageError("invalid conversation id")
        return conversation_id

    @classmethod
    def session_filename(cls, conversation_id: str) -> str:
        return f"{cls.validate_id(conversation_id)}.bin"

    def create_conversation(self, title: str = "New conversation") -> dict[str, Any]:
        conversation_id = secrets.token_hex(16)
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title[:120], now, now),
            )
        return self.get_conversation(conversation_id) or {}

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT c.*, (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def messages(self, conversation_id: str) -> list[dict[str, str]]:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content, reasoning_content FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"],
             **({"reasoning_content": row["reasoning_content"]} if row["reasoning_content"] else {})}
            for row in rows
        ]

    def display_messages(self, conversation_id: str) -> list[dict[str, str]]:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT role, coalesce(display_content, content) AS content, reasoning_content FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"],
             **({"reasoning_content": row["reasoning_content"]} if row["reasoning_content"] else {})}
            for row in rows
        ]

    def append_exchange(self, conversation_id: str, user_text: str, assistant_text: str,
                        image_prompt: str = "", aspect_ratio: str = "1:1 (Square)",
                        display_answer: str | None = None, reasoning_content: str = "") -> None:
        conversation_id = self.validate_id(conversation_id)
        now = _now()
        with self._lock, self._connect() as connection:
            existing = connection.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if existing is None:
                raise StorageError("conversation does not exist")
            connection.executemany(
                                "INSERT INTO messages (conversation_id, role, content, display_content, reasoning_content, created_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                ((conversation_id, "user", user_text, user_text, "", now),
                                 (conversation_id, "assistant", assistant_text, display_answer or assistant_text,
                                    reasoning_content, now)),
            )
            title = existing["title"]
            if title == "New conversation":
                title = user_text.strip().replace("\n", " ")[:72] or title
            connection.execute("UPDATE conversations SET title = ?, image_prompt = ?, aspect_ratio = ?, updated_at = ? "
                               "WHERE id = ?", (title, image_prompt, aspect_ratio, now, conversation_id))

    def mark_context_saved(self, conversation_id: str, token_count: int, byte_count: int) -> None:
        conversation_id = self.validate_id(conversation_id)
        if token_count < 0 or byte_count < 0:
            raise StorageError("saved context sizes must be non-negative")
        with self._lock, self._connect() as connection:
            message_count = connection.execute(
                "SELECT count(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            cursor = connection.execute(
                "UPDATE conversations SET context_saved_at = ?, context_tokens = ?, context_bytes = ?, "
                "context_message_count = ?, updated_at = ? WHERE id = ?",
                (_now(), token_count, byte_count, message_count, _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise StorageError("conversation does not exist")

    def context_is_current(self, conversation_id: str) -> bool:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT c.context_saved_at, c.context_message_count, "
                "(SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS current_count "
                "FROM conversations c WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
        return bool(row and row["context_saved_at"] and row["context_message_count"] == row["current_count"])

    def active_conversation_id(self) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT value FROM app_state WHERE key = 'strata_conversation_id'").fetchone()
        if row is None:
            return None
        return self.validate_id(row["value"])

    def set_active_conversation_id(self, conversation_id: str | None) -> None:
        if conversation_id is not None:
            conversation_id = self.validate_id(conversation_id)
            if self.get_conversation(conversation_id) is None:
                raise StorageError("conversation does not exist")
        with self._lock, self._connect() as connection:
            if conversation_id is None:
                connection.execute("DELETE FROM app_state WHERE key = 'strata_conversation_id'")
            else:
                connection.execute(
                    "INSERT INTO app_state (key, value) VALUES ('strata_conversation_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (conversation_id,),
                )

    def selected_conversation_id(self) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT value FROM app_state WHERE key = 'selected_conversation_id'").fetchone()
        if row is None:
            return None
        return self.validate_id(row["value"])

    def set_selected_conversation_id(self, conversation_id: str | None) -> None:
        if conversation_id is not None:
            conversation_id = self.validate_id(conversation_id)
            if self.get_conversation(conversation_id) is None:
                raise StorageError("conversation does not exist")
        with self._lock, self._connect() as connection:
            if conversation_id is None:
                connection.execute("DELETE FROM app_state WHERE key = 'selected_conversation_id'")
            else:
                connection.execute(
                    "INSERT INTO app_state (key, value) VALUES ('selected_conversation_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (conversation_id,),
                )

    def set_conversation_mode(self, conversation_id: str, image_mode: str) -> None:
        conversation_id = self.validate_id(conversation_id)
        if image_mode not in {"text-to-image", "edit"}:
            raise StorageError("image mode must be text-to-image or edit")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET image_mode = ?, updated_at = ? WHERE id = ?",
                (image_mode, _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise StorageError("conversation does not exist")

    @staticmethod
    def validate_source_image_id(source_image_id: str) -> str:
        if not isinstance(source_image_id, str) or not _SOURCE_IMAGE_ID.fullmatch(source_image_id):
            raise StorageError("invalid source image id")
        return source_image_id

    def create_source_image(self, source_image_id: str, conversation_id: str, filename: str,
                            subfolder: str, content_type: str, size_bytes: int,
                            width: int, height: int) -> dict[str, Any]:
        source_image_id = self.validate_source_image_id(source_image_id)
        conversation_id = self.validate_id(conversation_id)
        if (not isinstance(filename, str) or filename in {"", ".", ".."} or Path(filename).name != filename
                or "\\" in filename or subfolder != conversation_id):
            raise StorageError("invalid ComfyUI source image path")
        if (not isinstance(content_type, str) or not content_type.startswith("image/")
                or isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0
                or isinstance(width, bool) or not isinstance(width, int) or width <= 0
                or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
            raise StorageError("invalid source image metadata")
        with self._lock, self._connect() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise StorageError("conversation does not exist")
            position = connection.execute(
                "SELECT coalesce(max(position), -1) + 1 FROM source_images WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO source_images "
                "(id, conversation_id, filename, subfolder, content_type, size_bytes, width, height, position, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source_image_id, conversation_id, filename, subfolder, content_type,
                 size_bytes, width, height, position, _now()),
            )
        return self.source_image(conversation_id, source_image_id) or {}

    def source_images(self, conversation_id: str) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT s.*, EXISTS (SELECT 1 FROM generation_sources g "
                "WHERE g.source_image_id = s.id) AS in_use FROM source_images s "
                "WHERE s.conversation_id = ? ORDER BY s.position, s.created_at",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def source_image(self, conversation_id: str, source_image_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        source_image_id = self.validate_source_image_id(source_image_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT s.*, EXISTS (SELECT 1 FROM generation_sources g "
                "WHERE g.source_image_id = s.id) AS in_use FROM source_images s "
                "WHERE s.id = ? AND s.conversation_id = ?",
                (source_image_id, conversation_id),
            ).fetchone()
        return dict(row) if row else None

    def source_images_by_ids(self, conversation_id: str,
                             source_image_ids: list[str]) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        if not 1 <= len(source_image_ids) <= 5:
            raise StorageError("edit mode requires one base image and at most four reference images")
        ids = [self.validate_source_image_id(source_id) for source_id in source_image_ids]
        if len(set(ids)) != len(ids):
            raise StorageError("source image ids must be unique")
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM source_images WHERE conversation_id = ? AND id IN ({placeholders})",
                (conversation_id, *ids),
            ).fetchall()
        records = {row["id"]: dict(row) for row in rows}
        if len(records) != len(ids):
            raise StorageError("source image does not belong to this conversation")
        return [records[source_id] for source_id in ids]

    def reorder_source_images(self, conversation_id: str, source_image_ids: list[str]) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        ids = [self.validate_source_image_id(source_id) for source_id in source_image_ids]
        if len(ids) > 5 or len(set(ids)) != len(ids):
            raise StorageError("source image order must contain at most five unique ids")
        with self._lock, self._connect() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise StorageError("conversation does not exist")
            rows = connection.execute(
                "SELECT id FROM source_images WHERE conversation_id = ? ORDER BY position, created_at",
                (conversation_id,),
            ).fetchall()
            current_ids = [row["id"] for row in rows]
            if len(current_ids) != len(ids) or set(current_ids) != set(ids):
                raise StorageError("source image order must include every image in this conversation")
            connection.executemany(
                "UPDATE source_images SET position = ? WHERE conversation_id = ? AND id = ?",
                ((position, conversation_id, source_id) for position, source_id in enumerate(ids)),
            )
        return self.source_images(conversation_id)

    def delete_source_image(self, conversation_id: str, source_image_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        source_image_id = self.validate_source_image_id(source_image_id)
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM source_images WHERE id = ? AND conversation_id = ?",
                    (source_image_id, conversation_id),
                ).fetchone()
                if row is None:
                    return None
                record = dict(row)
                connection.execute(
                    "DELETE FROM source_images WHERE id = ? AND conversation_id = ?",
                    (source_image_id, conversation_id),
                )
        except sqlite3.IntegrityError as error:
            raise StorageConflictError("source image is used by a generation") from error
        return record

    def delete_conversation(self, conversation_id: str) -> bool:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM generation_sources WHERE generation_id IN "
                "(SELECT id FROM generations WHERE conversation_id = ?)",
                (conversation_id,),
            )
            cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            if cursor.rowcount != 1:
                return False
            connection.execute("DELETE FROM app_state WHERE value = ?", (conversation_id,))
        return True

    def create_generation(self, conversation_id: str, prompt: str, seed: int, aspect_ratio: str,
                          megapixels: float, width: int, height: int, sampler_name: str,
                          scheduler: str, image_mode: str = "text-to-image",
                          source_image_ids: list[str] | None = None) -> str:
        conversation_id = self.validate_id(conversation_id)
        if image_mode not in {"text-to-image", "edit"}:
            raise StorageError("image mode must be text-to-image or edit")
        source_ids = source_image_ids or []
        if image_mode == "edit":
            if not 1 <= len(source_ids) <= 5:
                raise StorageError("edit mode requires one base image and at most four reference images")
            source_ids = [self.validate_source_image_id(source_id) for source_id in source_ids]
            if len(set(source_ids)) != len(source_ids):
                raise StorageError("source image ids must be unique")
        elif source_ids:
            raise StorageError("text-to-image mode does not accept source images")
        generation_id = secrets.token_hex(16)
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO generations (id, conversation_id, prompt, seed, aspect_ratio, megapixels, width, height, "
                "sampler_name, scheduler, status, image_mode, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                (generation_id, conversation_id, prompt, seed, aspect_ratio, megapixels, width, height,
                 sampler_name, scheduler, image_mode, now, now),
            )
            for position, source_image_id in enumerate(source_ids):
                cursor = connection.execute(
                    "INSERT INTO generation_sources (generation_id, source_image_id, position) "
                    "SELECT ?, id, ? FROM source_images WHERE id = ? AND conversation_id = ?",
                    (generation_id, position, source_image_id, conversation_id),
                )
                if cursor.rowcount != 1:
                    raise StorageError("source image does not belong to this conversation")
        return generation_id

    def update_generation(self, generation_id: str, status: str, prompt_id: str | None = None,
                          image: dict[str, str] | None = None, error: str | None = None) -> None:
        allowed = {"queued", "running", "complete", "failed"}
        if status not in allowed:
            raise StorageError("invalid generation status")
        values = image or {}
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE generations SET status = ?, prompt_id = coalesce(?, prompt_id), filename = coalesce(?, filename), "
                "subfolder = coalesce(?, subfolder), image_type = coalesce(?, image_type), error = ?, updated_at = ? "
                "WHERE id = ?",
                (status, prompt_id, values.get("filename"), values.get("subfolder"), values.get("type"),
                 error, _now(), generation_id),
            )
            if cursor.rowcount != 1:
                raise StorageError("generation does not exist")

    def generations(self, conversation_id: str) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM generations WHERE conversation_id = ? ORDER BY created_at DESC",
                (conversation_id,),
            ).fetchall()
            return [self._generation_with_sources(connection, row) for row in rows]

    def generation(self, generation_id: str) -> dict[str, Any] | None:
        if not isinstance(generation_id, str) or not re.fullmatch(r"^[0-9a-f]{32}$", generation_id):
            raise StorageError("invalid generation id")
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM generations WHERE id = ?", (generation_id,)).fetchone()
            return self._generation_with_sources(connection, row) if row else None

    @staticmethod
    def _generation_with_sources(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        generation = dict(row)
        sources = connection.execute(
            "SELECT source_image_id FROM generation_sources WHERE generation_id = ? ORDER BY position",
            (generation["id"],),
        ).fetchall()
        generation["source_image_ids"] = [source["source_image_id"] for source in sources]
        return generation