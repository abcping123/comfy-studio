from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from comfy_studio.storage import ConversationStore, StorageError


def _assert_private_file(test_case: unittest.TestCase, path: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["icacls", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        entries = [line.strip() for line in result.stdout.splitlines() if ":(" in line]
        test_case.assertEqual(len(entries), 1)
        test_case.assertTrue(entries[0].endswith(":(F)"))
    else:
        test_case.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / "temp"
        self.store = ConversationStore(root / "session", root / "runtime")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_conversation_messages_and_native_context_metadata_persist(self) -> None:
        conversation = self.store.create_conversation()
        conversation_id = conversation["id"]
        self.store.append_exchange(conversation_id, "A red teapot", "I prepared the image prompt.")
        self.store.mark_context_saved(
            conversation_id, 128, 4096, backend="strata",
        )
        self.store.set_active_conversation_id(conversation_id)
        self.store.set_selected_conversation_id(conversation_id)
        self.assertEqual(self.store.messages(conversation_id), [
            {"role": "user", "content": "A red teapot"},
            {"role": "assistant", "content": "I prepared the image prompt."},
        ])
        saved = self.store.get_conversation(conversation_id)
        self.assertEqual(saved["context_tokens"], 128)
        self.assertEqual(saved["context_bytes"], 4096)
        self.assertEqual(saved["context_backend"], "strata")
        self.assertTrue(self.store.context_is_current(conversation_id))
        self.assertEqual(self.store.active_conversation_id(), conversation_id)
        self.assertEqual(self.store.selected_conversation_id(), conversation_id)
        self.assertEqual(
            self.store.session_filename(conversation_id),
            f"strata_context_{conversation_id}.bin",
        )
        _assert_private_file(
            self, self.store.session_root / conversation_id / "session.sqlite3",
        )
        _assert_private_file(self, self.store.runtime_root / "state.json")

    def test_context_snapshot_is_stale_after_new_messages(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.store.append_exchange(conversation_id, "First question", "First answer")
        self.store.mark_context_saved(
            conversation_id, 12, 256, backend="strata",
        )
        self.assertTrue(self.store.context_is_current(conversation_id))
        self.store.append_exchange(conversation_id, "Second question", "Second answer")
        self.assertFalse(self.store.context_is_current(conversation_id))

    def test_context_snapshot_is_scoped_to_its_backend(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.store.append_exchange(conversation_id, "Question", "Answer")
        self.store.mark_context_saved(conversation_id, None, 128, backend="bonsai")

        self.assertTrue(self.store.context_is_current(conversation_id, "bonsai"))
        self.assertFalse(self.store.context_is_current(conversation_id, "strata"))
        self.assertEqual(
            self.store.get_conversation(conversation_id)["context_backend"],
            "bonsai",
        )

    def test_raw_transcript_and_normalized_display_text_are_separate(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        raw = '{"reply":"ok","aspect_ratio":"5:7"}'
        display = "ok (using the nearest supported ratio)"
        reasoning = "Plan the composition before writing the prompt."
        self.store.append_exchange(conversation_id, "Make a poster", raw, "Prompt",
                                   "3:4 (Portrait Standard)", display, reasoning)
        self.assertEqual(self.store.messages(conversation_id)[-1]["content"], raw)
        self.assertEqual(self.store.messages(conversation_id)[-1]["reasoning_content"], reasoning)
        self.assertEqual(self.store.display_messages(conversation_id)[-1]["content"], display)
        self.assertEqual(self.store.display_messages(conversation_id)[-1]["reasoning_content"], reasoning)

    def test_each_conversation_has_its_own_database_outside_runtime(self) -> None:
        first_id = self.store.create_conversation()["id"]
        second_id = self.store.create_conversation()["id"]

        first_database = self.store.session_root / first_id / "session.sqlite3"
        second_database = self.store.session_root / second_id / "session.sqlite3"
        self.assertNotEqual(first_database, second_database)
        self.assertTrue(first_database.is_file())
        self.assertTrue(second_database.is_file())
        self.assertEqual(list(self.store.runtime_root.glob("*.sqlite3")), [])

    def test_generation_lifecycle_is_persisted(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        generation_id = self.store.create_generation(conversation_id, "A red teapot", 42,
                                "1:1 (Square)", 4.0, 2048, 2048, "euler", "simple")
        self.store.update_generation(generation_id, "running", prompt_id="prompt-1")
        self.store.update_generation(generation_id, "complete", image={
            "filename": "image.png", "subfolder": "", "type": "output",
        })
        generation = self.store.generations(conversation_id)[0]
        self.assertEqual(generation["status"], "complete")
        self.assertEqual(generation["prompt_id"], "prompt-1")
        self.assertEqual(generation["filename"], "image.png")
        self.assertEqual(generation["aspect_ratio"], "1:1 (Square)")
        self.assertEqual(generation["megapixels"], 4.0)
        self.assertEqual(generation["width"], 2048)
        self.assertEqual(generation["height"], 2048)
        self.assertEqual(generation["sampler_name"], "euler")
        self.assertEqual(generation["scheduler"], "simple")
        self.assertEqual(generation["image_mode"], "text-to-image")
        self.assertEqual(generation["source_image_ids"], [])

    def test_source_images_are_ordered_scoped_and_linked_to_edit_generations(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        other_conversation_id = self.store.create_conversation()["id"]
        self.store.set_conversation_mode(conversation_id, "edit")
        base_id = "1" * 32
        reference_id = "2" * 32
        self.store.create_source_image(
            base_id, conversation_id, f"{base_id}.png", conversation_id,
            "image/png", 100, 1920, 1080,
        )
        self.store.create_source_image(
            reference_id, conversation_id, f"{reference_id}.jpg", conversation_id,
            "image/jpeg", 200, 800, 1200,
        )

        generation_id = self.store.create_generation(
            conversation_id, "Edit <Picture 1>", 7, "Source image", 4.0,
            2720, 1504, "euler", "simple", "edit", [base_id, reference_id],
        )

        generation = self.store.generation(generation_id)
        self.assertEqual(generation["image_mode"], "edit")
        self.assertEqual(generation["source_image_ids"], [base_id, reference_id])
        self.assertTrue(self.store.source_images(conversation_id)[0]["in_use"])
        with self.assertRaisesRegex(StorageError, "does not belong"):
            self.store.source_images_by_ids(other_conversation_id, [base_id])
        with self.assertRaisesRegex(StorageError, "used by"):
            self.store.delete_source_image(conversation_id, base_id)

        reordered = self.store.reorder_source_images(conversation_id, [reference_id, base_id])
        self.assertEqual([image["id"] for image in reordered], [reference_id, base_id])
        self.assertTrue(self.store.delete_conversation(conversation_id))
        self.assertEqual(self.store.source_images(conversation_id), [])
        self.assertIsNone(self.store.get_conversation(conversation_id))

    def test_invalid_ids_and_nonexistent_conversations_are_rejected(self) -> None:
        with self.assertRaisesRegex(StorageError, "invalid conversation id"):
            self.store.messages("../outside")
        with self.assertRaisesRegex(StorageError, "does not exist"):
            self.store.append_exchange("0" * 32, "A prompt", "An answer")

    def test_delete_cascades_records_and_clears_conversation_state(self) -> None:
        conversation_id = self.store.create_conversation()["id"]
        self.store.append_exchange(conversation_id, "A prompt", "A reply")
        self.store.set_active_conversation_id(conversation_id)
        self.store.set_selected_conversation_id(conversation_id)
        generation_id = self.store.create_generation(conversation_id, "A prompt", 1,
                                                     "1:1 (Square)", 4.0, 2048, 2048, "euler", "simple")

        self.assertTrue(self.store.delete_conversation(conversation_id))

        self.assertIsNone(self.store.get_conversation(conversation_id))
        self.assertEqual(self.store.messages(conversation_id), [])
        self.assertEqual(self.store.generations(conversation_id), [])
        self.assertIsNone(self.store.active_conversation_id())
        self.assertIsNone(self.store.selected_conversation_id())
        self.assertIsNone(self.store.generation(generation_id))
        self.assertFalse(self.store.delete_conversation(conversation_id))


if __name__ == "__main__":
    unittest.main()