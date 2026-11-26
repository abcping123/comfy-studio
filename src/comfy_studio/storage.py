from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chat_backends import CHAT_BACKENDS
from .modes import DEFAULT_MODE_ID, MAX_SOURCE_IMAGES, get_mode, validate_source_image_count
from .permissions import set_private_permissions


_CONVERSATION_ID = re.compile(r"^[0-9a-f]{32}$")
_SOURCE_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")
_CONTENT_HASH = re.compile(r"^[0-9a-f]{64}$")


class StorageError(RuntimeError):
    """Persistent app state is missing or invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, session_root: Path, runtime_root: Path):
        self.session_root = Path(session_root)
        self.runtime_root = Path(runtime_root)
        self._state_path = self.runtime_root / "state.json"
        self._lock = threading.RLock()
        self._initialized: set[str] = set()
        self._summary_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}
        self._generation_conversations: dict[str, str] = {}
        self._prepare_directory(self.session_root)
        self._prepare_directory(self.runtime_root)

    @staticmethod
    def _prepare_directory(path: Path) -> None:
        if path.is_symlink():
            raise StorageError(f"refusing to use a symbolic-link storage directory: {path}")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not path.is_dir():
            raise StorageError(f"storage path is not a directory: {path}")
        set_private_permissions(path, directory=True)

    def session_directory(self, conversation_id: str, *, create: bool = False) -> Path | None:
        conversation_id = self.validate_id(conversation_id)
        if self.session_root.is_symlink():
            raise StorageError("refusing to use a symbolic-link session root")
        if not self.session_root.exists():
            if not create:
                return None
            self._prepare_directory(self.session_root)
        root = self.session_root.resolve(strict=True)
        directory = root / conversation_id
        if directory.is_symlink():
            raise StorageError("refusing to use a symbolic-link session directory")
        if not directory.exists():
            if not create:
                return None
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError as error:
                raise StorageError("session directory already exists") from error
            set_private_permissions(directory, directory=True)
        if not directory.is_dir():
            raise StorageError("session path is not a directory")
        resolved = directory.resolve(strict=True)
        if resolved.parent != root:
            raise StorageError("session directory is outside the configured session root")
        return resolved

    def database_path(self, conversation_id: str) -> Path | None:
        directory = self.session_directory(conversation_id)
        if directory is None:
            return None
        path = directory / "session.sqlite3"
        if path.is_symlink():
            raise StorageError("refusing to use a symbolic-link session database")
        if path.exists() and not path.is_file():
            raise StorageError("session database path is not a file")
        return path if path.is_file() else None

    @contextmanager
    def _connect(self, conversation_id: str) -> Iterator[sqlite3.Connection]:
        conversation_id = self.validate_id(conversation_id)
        path = self.database_path(conversation_id)
        if path is None:
            raise StorageError("conversation database does not exist")
        if conversation_id not in self._initialized:
            self._initialize(conversation_id)
        connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self, conversation_id: str) -> None:
        path = self.database_path(conversation_id)
        if path is None:
            raise StorageError("conversation database does not exist")
        with sqlite3.connect(path, timeout=30) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(f"""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    context_saved_at TEXT,
                    context_tokens INTEGER,
                    context_bytes INTEGER,
                    context_message_count INTEGER,
                    context_backend TEXT,
                    image_prompt TEXT NOT NULL DEFAULT '',
                    aspect_ratio TEXT NOT NULL DEFAULT '1:1 (Square)',
                    image_mode TEXT NOT NULL DEFAULT '{DEFAULT_MODE_ID}'
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
                    image_mode TEXT NOT NULL DEFAULT '{DEFAULT_MODE_ID}',
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
                    content_hash TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    removed_at TEXT
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
            """)
        set_private_permissions(path)
        self._initialized.add(conversation_id)

    @staticmethod
    def validate_id(conversation_id: str) -> str:
        if not isinstance(conversation_id, str) or not _CONVERSATION_ID.fullmatch(conversation_id):
            raise StorageError("invalid conversation id")
        return conversation_id

    @classmethod
    def session_filename(cls, conversation_id: str) -> str:
        return f"strata_context_{cls.validate_id(conversation_id)}.bin"

    def create_conversation(self, title: str = "New conversation") -> dict[str, Any]:
        conversation_id = secrets.token_hex(16)
        now = _now()
        with self._lock:
            directory = self.session_directory(conversation_id, create=True)
            if directory is None:
                raise StorageError("could not create session directory")
            path = directory / "session.sqlite3"
            try:
                with path.open("xb"):
                    pass
                set_private_permissions(path)
                self._initialize(conversation_id)
                with self._connect(conversation_id) as connection:
                    connection.execute(
                        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (conversation_id, title[:120], now, now),
                    )
            except (OSError, sqlite3.Error, StorageError) as error:
                self._initialized.discard(conversation_id)
                try:
                    shutil.rmtree(directory)
                except OSError as cleanup_error:
                    raise StorageError(
                        f"could not initialize a session, and its directory could not be removed: {cleanup_error}"
                    ) from cleanup_error
                if isinstance(error, StorageError):
                    raise
                raise StorageError(f"could not initialize session storage: {error}") from error
        return self.get_conversation(conversation_id) or {}

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        if self.database_path(conversation_id) is None:
            return None
        with self._lock, self._connect(conversation_id) as connection:
            row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None

    def _invalidate_summary(self, conversation_id: str) -> None:
        self._summary_cache.pop(conversation_id, None)

    def _conversation_summary(self, conversation_id: str) -> dict[str, Any]:
        path = self.database_path(conversation_id)
        if path is None:
            self._summary_cache.pop(conversation_id, None)
            raise StorageError(f"session directory {conversation_id} has no conversation database")
        try:
            stat_result = path.stat()
        except OSError as error:
            raise StorageError(f"could not stat session database: {error}") from error
        signature = (stat_result.st_mtime_ns, stat_result.st_size)
        cached = self._summary_cache.get(conversation_id)
        if cached is not None and cached[:2] == signature:
            return dict(cached[2])
        with self._connect(conversation_id) as connection:
            row = connection.execute(
                "SELECT c.*, (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) "
                "AS message_count FROM conversations c WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            self._summary_cache.pop(conversation_id, None)
            raise StorageError(f"session directory {conversation_id} has no conversation record")
        summary = dict(row)
        try:
            post_stat = path.stat()
            self._summary_cache[conversation_id] = (
                post_stat.st_mtime_ns, post_stat.st_size, dict(summary),
            )
        except OSError:
            self._summary_cache.pop(conversation_id, None)
        return summary

    def list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        with self._lock:
            session_ids = self._session_ids()
            active_ids = set(session_ids)
            for cached_id in list(self._summary_cache):
                if cached_id not in active_ids:
                    del self._summary_cache[cached_id]
            for conversation_id in session_ids:
                conversations.append(self._conversation_summary(conversation_id))
        return sorted(conversations, key=lambda row: row["updated_at"], reverse=True)

    def _session_ids(self) -> list[str]:
        if self.session_root.is_symlink():
            raise StorageError("refusing to scan a symbolic-link session root")
        try:
            entries = list(self.session_root.iterdir())
        except OSError as error:
            raise StorageError(f"could not scan session directories: {error}") from error
        ids: list[str] = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                raise StorageError(f"unexpected file in session root: {entry.name}")
            try:
                conversation_id = self.validate_id(entry.name)
            except StorageError as error:
                database_path = entry / "session.sqlite3"
                if database_path.exists() or database_path.is_symlink():
                    raise StorageError(
                        f"session directory has an invalid conversation id: {entry.name}"
                    ) from error
                continue
            ids.append(conversation_id)
        return ids

    def messages(self, conversation_id: str) -> list[dict[str, str]]:
        conversation_id = self.validate_id(conversation_id)
        if self.database_path(conversation_id) is None:
            return []
        with self._lock, self._connect(conversation_id) as connection:
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
        if self.database_path(conversation_id) is None:
            return []
        with self._lock, self._connect(conversation_id) as connection:
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
        with self._lock, self._connect(conversation_id) as connection:
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
            self._invalidate_summary(conversation_id)

    def mark_context_saved(self, conversation_id: str, token_count: int | None,
                           byte_count: int, backend: str) -> None:
        conversation_id = self.validate_id(conversation_id)
        if backend not in CHAT_BACKENDS:
            raise StorageError(
                f"saved context backend must be one of: {', '.join(CHAT_BACKENDS)}"
            )
        if (token_count is not None and token_count < 0) or byte_count < 0:
            raise StorageError("saved context sizes must be non-negative")
        with self._lock, self._connect(conversation_id) as connection:
            message_count = connection.execute(
                "SELECT count(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            cursor = connection.execute(
                "UPDATE conversations SET context_saved_at = ?, context_tokens = ?, context_bytes = ?, "
                "context_message_count = ?, context_backend = ?, updated_at = ? WHERE id = ?",
                (_now(), token_count, byte_count, message_count, backend, _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise StorageError("conversation does not exist")
            self._invalidate_summary(conversation_id)

    def context_is_current(self, conversation_id: str, backend: str | None = None) -> bool:
        conversation_id = self.validate_id(conversation_id)
        if self.database_path(conversation_id) is None:
            return False
        with self._lock, self._connect(conversation_id) as connection:
            row = connection.execute(
                "SELECT c.context_saved_at, c.context_message_count, c.context_backend, "
                "(SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS current_count "
                "FROM conversations c WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
        return bool(
            row
            and row["context_saved_at"]
            and row["context_message_count"] == row["current_count"]
            and (backend is None or row["context_backend"] == backend)
        )

    def _read_state(self) -> dict[str, Any]:
        if self._state_path.is_symlink():
            raise StorageError("refusing to read a symbolic-link runtime state file")
        if not self._state_path.exists():
            return {}
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StorageError(f"could not read runtime state: {error}") from error
        if not isinstance(state, dict):
            raise StorageError("runtime state must be a JSON object")
        for key in ("active_conversation_id", "selected_conversation_id"):
            value = state.get(key)
            if value is not None:
                try:
                    self.validate_id(value)
                except StorageError as error:
                    raise StorageError(f"runtime state contains an invalid {key}") from error
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        if self._state_path.is_symlink():
            raise StorageError("refusing to write a symbolic-link runtime state file")
        temporary = self.runtime_root / f".state-{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(state, file, ensure_ascii=False, separators=(",", ":"))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._state_path)
            set_private_permissions(self._state_path)
        except OSError as error:
            raise StorageError(f"could not write runtime state: {error}") from error
        finally:
            temporary.unlink(missing_ok=True)

    def _state_value(self, key: str) -> str | None:
        with self._lock:
            state = self._read_state()
            changed = False
            for state_key in ("active_conversation_id", "selected_conversation_id"):
                conversation_id = state.get(state_key)
                if conversation_id is not None and self.database_path(conversation_id) is None:
                    state.pop(state_key)
                    changed = True
            if changed:
                self._write_state(state)
            return state.get(key)

    def _set_state_value(self, key: str, conversation_id: str | None) -> None:
        if conversation_id is not None:
            conversation_id = self.validate_id(conversation_id)
            if self.get_conversation(conversation_id) is None:
                raise StorageError("conversation does not exist")
        with self._lock:
            state = self._read_state()
            if conversation_id is None:
                state.pop(key, None)
            else:
                state[key] = conversation_id
            self._write_state(state)

    def active_conversation_id(self) -> str | None:
        return self._state_value("active_conversation_id")

    def set_active_conversation_id(self, conversation_id: str | None) -> None:
        self._set_state_value("active_conversation_id", conversation_id)

    def selected_conversation_id(self) -> str | None:
        return self._state_value("selected_conversation_id")

    def set_selected_conversation_id(self, conversation_id: str | None) -> None:
        self._set_state_value("selected_conversation_id", conversation_id)

    def set_conversation_mode(self, conversation_id: str, image_mode: str) -> None:
        conversation_id = self.validate_id(conversation_id)
        try:
            get_mode(image_mode)
        except ValueError as error:
            raise StorageError(str(error)) from error
        with self._lock, self._connect(conversation_id) as connection:
            cursor = connection.execute(
                "UPDATE conversations SET image_mode = ?, updated_at = ? WHERE id = ?",
                (image_mode, _now(), conversation_id),
            )
            if cursor.rowcount != 1:
                raise StorageError("conversation does not exist")
            self._invalidate_summary(conversation_id)

    @staticmethod
    def validate_source_image_id(source_image_id: str) -> str:
        if not isinstance(source_image_id, str) or not _SOURCE_IMAGE_ID.fullmatch(source_image_id):
            raise StorageError("invalid source image id")
        return source_image_id

    def create_source_image(self, source_image_id: str, conversation_id: str, filename: str,
                            subfolder: str, content_type: str, content_hash: str,
                            size_bytes: int, width: int, height: int) -> dict[str, Any]:
        source_image_id = self.validate_source_image_id(source_image_id)
        conversation_id = self.validate_id(conversation_id)
        if not isinstance(content_hash, str) or not _CONTENT_HASH.fullmatch(content_hash):
            raise StorageError("invalid source image content hash")
        if (not isinstance(content_type, str) or not content_type.startswith("image/")
                or isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0
                or isinstance(width, bool) or not isinstance(width, int) or width <= 0
                or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
            raise StorageError("invalid source image metadata")
        extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(content_type)
        if (extension is None or filename != f"{content_hash}{extension}"
                or subfolder != conversation_id):
            raise StorageError("invalid ComfyUI source image path")
        with self._lock, self._connect(conversation_id) as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise StorageError("conversation does not exist")
            position = connection.execute(
                "SELECT coalesce(max(position), -1) + 1 FROM source_images "
                "WHERE conversation_id = ? AND removed_at IS NULL",
                (conversation_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO source_images "
                "(id, conversation_id, filename, subfolder, content_type, content_hash, size_bytes, "
                "width, height, position, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source_image_id, conversation_id, filename, subfolder, content_type, content_hash,
                 size_bytes, width, height, position, _now()),
            )
        return self.source_image(conversation_id, source_image_id) or {}

    def source_images(self, conversation_id: str) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        if self.database_path(conversation_id) is None:
            return []
        with self._lock, self._connect(conversation_id) as connection:
            rows = connection.execute(
                "SELECT * FROM source_images "
                "WHERE conversation_id = ? AND removed_at IS NULL ORDER BY position, created_at",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def source_image(self, conversation_id: str, source_image_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        source_image_id = self.validate_source_image_id(source_image_id)
        if self.database_path(conversation_id) is None:
            return None
        with self._lock, self._connect(conversation_id) as connection:
            row = connection.execute(
                "SELECT * FROM source_images "
                "WHERE id = ? AND conversation_id = ? AND removed_at IS NULL",
                (source_image_id, conversation_id),
            ).fetchone()
        return dict(row) if row else None

    def source_image_by_hash(self, conversation_id: str, content_hash: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        if not isinstance(content_hash, str) or not _CONTENT_HASH.fullmatch(content_hash):
            raise StorageError("invalid source image content hash")
        if self.database_path(conversation_id) is None:
            return None
        with self._lock, self._connect(conversation_id) as connection:
            row = connection.execute(
                "SELECT * FROM source_images WHERE conversation_id = ? AND content_hash = ?",
                (conversation_id, content_hash),
            ).fetchone()
        return dict(row) if row else None

    def restore_source_image(self, conversation_id: str, source_image_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        source_image_id = self.validate_source_image_id(source_image_id)
        if self.database_path(conversation_id) is None:
            return None
        with self._lock, self._connect(conversation_id) as connection:
            row = connection.execute(
                "SELECT id, removed_at FROM source_images WHERE id = ? AND conversation_id = ?",
                (source_image_id, conversation_id),
            ).fetchone()
            if row is None:
                return None
            if row["removed_at"] is not None:
                position = connection.execute(
                    "SELECT coalesce(max(position), -1) + 1 FROM source_images "
                    "WHERE conversation_id = ? AND removed_at IS NULL",
                    (conversation_id,),
                ).fetchone()[0]
                connection.execute(
                    "UPDATE source_images SET removed_at = NULL, position = ? "
                    "WHERE id = ? AND conversation_id = ?",
                    (position, source_image_id, conversation_id),
                )
        return self.source_image(conversation_id, source_image_id)

    def source_images_by_ids(self, conversation_id: str,
                             source_image_ids: list[str]) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        if len(source_image_ids) > MAX_SOURCE_IMAGES:
            raise StorageError(f"source image ids must contain at most {MAX_SOURCE_IMAGES} ids")
        ids = [self.validate_source_image_id(source_id) for source_id in source_image_ids]
        if len(set(ids)) != len(ids):
            raise StorageError("source image ids must be unique")
        if not ids:
            return []
        if self.database_path(conversation_id) is None:
            raise StorageError("source image does not belong to this conversation")
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect(conversation_id) as connection:
            rows = connection.execute(
                f"SELECT * FROM source_images WHERE conversation_id = ? "
                f"AND removed_at IS NULL AND id IN ({placeholders})",
                (conversation_id, *ids),
            ).fetchall()
        records = {row["id"]: dict(row) for row in rows}
        if len(records) != len(ids):
            raise StorageError("source image does not belong to this conversation")
        return [records[source_id] for source_id in ids]

    def reorder_source_images(self, conversation_id: str, source_image_ids: list[str]) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        ids = [self.validate_source_image_id(source_id) for source_id in source_image_ids]
        if len(ids) > MAX_SOURCE_IMAGES or len(set(ids)) != len(ids):
            raise StorageError(
                f"source image order must contain at most {MAX_SOURCE_IMAGES} unique ids"
            )
        with self._lock, self._connect(conversation_id) as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise StorageError("conversation does not exist")
            rows = connection.execute(
                "SELECT id FROM source_images WHERE conversation_id = ? AND removed_at IS NULL "
                "ORDER BY position, created_at",
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

    def soft_delete_source_image(self, conversation_id: str,
                                 source_image_id: str) -> dict[str, Any] | None:
        conversation_id = self.validate_id(conversation_id)
        source_image_id = self.validate_source_image_id(source_image_id)
        if self.database_path(conversation_id) is None:
            return None
        removed_at = _now()
        with self._lock, self._connect(conversation_id) as connection:
            cursor = connection.execute(
                "UPDATE source_images SET removed_at = ? "
                "WHERE id = ? AND conversation_id = ? AND removed_at IS NULL",
                (removed_at, source_image_id, conversation_id),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM source_images WHERE id = ? AND conversation_id = ?",
                (source_image_id, conversation_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_conversation(self, conversation_id: str) -> bool:
        conversation_id = self.validate_id(conversation_id)
        with self._lock:
            directory = self.session_directory(conversation_id)
            if directory is None or self.get_conversation(conversation_id) is None:
                return False
            state = self._read_state()
            updated = {
                key: value for key, value in state.items()
                if value != conversation_id
            }
            if updated != state:
                self._write_state(updated)
            try:
                shutil.rmtree(directory)
            except OSError as error:
                raise StorageError(f"could not remove session directory: {error}") from error
            self._initialized.discard(conversation_id)
            self._invalidate_summary(conversation_id)
            for gen_id, owner_id in list(self._generation_conversations.items()):
                if owner_id == conversation_id:
                    del self._generation_conversations[gen_id]
            return True

    def create_generation(self, conversation_id: str, prompt: str, seed: int, aspect_ratio: str,
                          megapixels: float, width: int, height: int, sampler_name: str,
                          scheduler: str, image_mode: str = DEFAULT_MODE_ID,
                          source_image_ids: list[str] | None = None) -> str:
        conversation_id = self.validate_id(conversation_id)
        try:
            mode = get_mode(image_mode)
        except ValueError as error:
            raise StorageError(str(error)) from error
        source_ids = source_image_ids or []
        try:
            validate_source_image_count(mode, len(source_ids))
        except ValueError as error:
            raise StorageError(str(error)) from error
        if source_ids:
            source_ids = [self.validate_source_image_id(source_id) for source_id in source_ids]
            if len(set(source_ids)) != len(source_ids):
                raise StorageError("source image ids must be unique")
        generation_id = secrets.token_hex(16)
        now = _now()
        with self._lock, self._connect(conversation_id) as connection:
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
            self._generation_conversations[generation_id] = conversation_id
        return generation_id

    def _candidate_generation_conversations(self, generation_id: str) -> list[str]:
        cached_id = self._generation_conversations.get(generation_id)
        session_ids = self._session_ids()
        if cached_id and cached_id in session_ids:
            return [cached_id, *[cid for cid in session_ids if cid != cached_id]]
        return session_ids

    def update_generation(self, generation_id: str, status: str, prompt_id: str | None = None,
                          image: dict[str, str] | None = None, error: str | None = None) -> None:
        allowed = {"queued", "running", "complete", "failed"}
        if status not in allowed:
            raise StorageError("invalid generation status")
        values = image or {}
        with self._lock:
            for conversation_id in self._candidate_generation_conversations(generation_id):
                with self._connect(conversation_id) as connection:
                    cursor = connection.execute(
                        "UPDATE generations SET status = ?, prompt_id = coalesce(?, prompt_id), "
                        "filename = coalesce(?, filename), subfolder = coalesce(?, subfolder), "
                        "image_type = coalesce(?, image_type), error = ?, updated_at = ? WHERE id = ?",
                        (status, prompt_id, values.get("filename"), values.get("subfolder"),
                         values.get("type"), error, _now(), generation_id),
                    )
                if cursor.rowcount == 1:
                    self._generation_conversations[generation_id] = conversation_id
                    return
        raise StorageError("generation does not exist")

    def generations(self, conversation_id: str) -> list[dict[str, Any]]:
        conversation_id = self.validate_id(conversation_id)
        if self.database_path(conversation_id) is None:
            return []
        with self._lock, self._connect(conversation_id) as connection:
            rows = connection.execute(
                "SELECT * FROM generations WHERE conversation_id = ? ORDER BY created_at DESC",
                (conversation_id,),
            ).fetchall()
            items = [self._generation_with_sources(connection, row) for row in rows]
            for item in items:
                self._generation_conversations[item["id"]] = conversation_id
            return items

    def generation(self, generation_id: str) -> dict[str, Any] | None:
        if not isinstance(generation_id, str) or not re.fullmatch(r"^[0-9a-f]{32}$", generation_id):
            raise StorageError("invalid generation id")
        with self._lock:
            for conversation_id in self._candidate_generation_conversations(generation_id):
                with self._connect(conversation_id) as connection:
                    row = connection.execute(
                        "SELECT * FROM generations WHERE id = ?", (generation_id,),
                    ).fetchone()
                    if row is not None:
                        self._generation_conversations[generation_id] = conversation_id
                        return self._generation_with_sources(connection, row)
        return None

    @staticmethod
    def _generation_with_sources(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        generation = dict(row)
        sources = connection.execute(
            "SELECT source_image_id FROM generation_sources WHERE generation_id = ? ORDER BY position",
            (generation["id"],),
        ).fetchall()
        generation["source_image_ids"] = [source["source_image_id"] for source in sources]
        return generation