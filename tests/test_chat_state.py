"""
Tests for `src.ui.chat_state`.

Acceptance:
  - append user, append assistant, replace last assistant, drop last work
    on empty AND non-empty lists without crashing.
  - mutations are in place (the same list reference is kept).
  - snapshot() returns a real copy.
  - container protocol (len, iter, bool, indexing) is a drop-in.
  - replace_last_assistant does NOT overwrite a user message — not even
    with `append_if_missing=False`.
"""

import unittest

from src.ui.chat_state import ChatState


class TestConstruction(unittest.TestCase):

    def test_empty(self):
        chat = ChatState()
        self.assertTrue(chat.is_empty())
        self.assertEqual(len(chat), 0)
        self.assertEqual(chat.snapshot(), [])

    def test_from_none(self):
        chat = ChatState(None)
        self.assertEqual(len(chat), 0)

    def test_from_list_keeps_reference(self):
        """Identity: ChatState works on the GIVEN list."""
        original = [{"role": "user", "content": "hi"}]
        chat = ChatState(original)
        self.assertIs(chat.messages, original)

    def test_from_iterable_makes_list(self):
        """For non-list input: copy."""
        gen = ({"role": "user", "content": str(i)} for i in range(2))
        chat = ChatState(gen)
        self.assertEqual(len(chat), 2)


class TestQueries(unittest.TestCase):

    def test_last_role_empty(self):
        self.assertIsNone(ChatState().last_role)

    def test_last_role(self):
        chat = ChatState([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        self.assertEqual(chat.last_role, "assistant")

    def test_last_content_empty(self):
        self.assertEqual(ChatState().last_content, "")

    def test_last_user_content(self):
        chat = ChatState([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "second"},
        ])
        self.assertEqual(chat.last_user_content, "second")

    def test_last_user_content_when_none(self):
        chat = ChatState([{"role": "assistant", "content": "x"}])
        self.assertIsNone(chat.last_user_content)

    def test_last_assistant_content_skips_intermediate_user(self):
        chat = ChatState([
            {"role": "assistant", "content": "first ans"},
            {"role": "user", "content": "follow up"},
        ])
        self.assertEqual(chat.last_assistant_content, "first ans")

    def test_has_pending_assistant(self):
        chat = ChatState([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ])
        self.assertTrue(chat.has_pending_assistant())
        self.assertFalse(chat.has_pending_user())

    def test_has_pending_assistant_empty(self):
        chat = ChatState()
        self.assertFalse(chat.has_pending_assistant())
        self.assertFalse(chat.has_pending_user())


class TestAppend(unittest.TestCase):

    def test_append_user(self):
        chat = ChatState()
        chat.append_user("hello")
        self.assertEqual(chat.messages, [
            {"role": "user", "content": "hello"},
        ])

    def test_append_assistant(self):
        chat = ChatState()
        chat.append_assistant("hi there")
        self.assertEqual(chat.messages, [
            {"role": "assistant", "content": "hi there"},
        ])

    def test_appends_mutate_original_list(self):
        original = []
        chat = ChatState(original)
        chat.append_user("x")
        # external view: the original list IS mutated
        self.assertEqual(len(original), 1)


class TestReplaceLastAssistant(unittest.TestCase):

    def test_replace_existing_assistant(self):
        chat = ChatState([{"role": "assistant", "content": "alt"}])
        chat.replace_last_assistant("neu")
        self.assertEqual(chat.last_assistant_content, "neu")

    def test_replace_when_empty_appends(self):
        """append_if_missing=True (default) appends."""
        chat = ChatState()
        chat.replace_last_assistant("erste Antwort")
        self.assertEqual(len(chat), 1)
        self.assertEqual(chat.last_assistant_content, "erste Antwort")

    def test_replace_when_last_is_user_appends(self):
        """A user message is NOT overwritten — it is kept."""
        chat = ChatState([{"role": "user", "content": "frage"}])
        chat.replace_last_assistant("antwort")
        self.assertEqual(len(chat), 2)
        # user message intact
        self.assertEqual(chat.last_user_content, "frage")
        self.assertEqual(chat.last_assistant_content, "antwort")

    def test_replace_no_append_safe_noop(self):
        """append_if_missing=False: no mutation with an empty list."""
        chat = ChatState()
        chat.replace_last_assistant("etwas", append_if_missing=False)
        self.assertEqual(len(chat), 0)

    def test_replace_keeps_extra_fields(self):
        """Extra fields in the dict (e.g. metadata) are kept."""
        chat = ChatState([{
            "role": "assistant",
            "content": "alt",
            "metadata": {"task": "explainer"},
        }])
        chat.replace_last_assistant("neu")
        self.assertEqual(
            chat.messages[0],
            {"role": "assistant", "content": "neu",
             "metadata": {"task": "explainer"}},
        )


class TestDropLast(unittest.TestCase):

    def test_drop_one(self):
        chat = ChatState([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        chat.drop_last()
        self.assertEqual(len(chat), 1)
        self.assertEqual(chat.last_role, "user")

    def test_drop_more_than_length(self):
        chat = ChatState([{"role": "user", "content": "x"}])
        chat.drop_last(5)
        self.assertEqual(len(chat), 0)

    def test_drop_zero_or_negative(self):
        chat = ChatState([{"role": "user", "content": "x"}])
        chat.drop_last(0)
        chat.drop_last(-1)
        self.assertEqual(len(chat), 1)

    def test_drop_n(self):
        chat = ChatState([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ])
        chat.drop_last(2)
        self.assertEqual(len(chat), 1)
        self.assertEqual(chat.last_user_content, "a")


class TestSnapshot(unittest.TestCase):

    def test_snapshot_is_copy(self):
        chat = ChatState([{"role": "user", "content": "x"}])
        snap = chat.snapshot()
        chat.append_assistant("y")
        # the snapshot is not mutated
        self.assertEqual(len(snap), 1)
        # but the ChatState itself is
        self.assertEqual(len(chat), 2)

    def test_snapshot_separate_list_objects(self):
        chat = ChatState([{"role": "user", "content": "x"}])
        self.assertIsNot(chat.snapshot(), chat.messages)


class TestContainerProtocol(unittest.TestCase):
    """Drop-in compatibility with a plain list."""

    def test_len(self):
        self.assertEqual(len(ChatState()), 0)
        self.assertEqual(len(ChatState([{}, {}])), 2)

    def test_bool(self):
        self.assertFalse(bool(ChatState()))
        self.assertTrue(bool(ChatState([{"role": "user", "content": "x"}])))

    def test_iter(self):
        msgs = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}]
        chat = ChatState(msgs)
        for got, want in zip(chat, msgs):
            self.assertEqual(got, want)

    def test_indexing(self):
        chat = ChatState([{"role": "user", "content": "a"}])
        self.assertEqual(chat[0]["role"], "user")
        self.assertEqual(chat[-1]["content"], "a")


class TestRealisticScenarios(unittest.TestCase):
    """Verification of the UI patterns from gradio_app.py."""

    def test_streaming_pattern(self):
        """Typical stream: user question, assistant placeholder, updates."""
        chat = ChatState()
        # Step 1: the user asks a question
        chat.append_user("Wie funktioniert X?")
        # Step 2: Assistant-Placeholder
        chat.append_assistant("🤔 *Thinking...*")
        # Step 3: Streaming-Updates
        chat.replace_last_assistant("X funktioniert so:...")
        chat.replace_last_assistant("X funktioniert so: zuerst Y, dann Z.")
        # final state
        self.assertEqual(len(chat), 2)
        self.assertEqual(
            chat.last_assistant_content,
            "X funktioniert so: zuerst Y, dann Z.",
        )

    def test_error_path_pattern(self):
        """Pipeline error: replace the last assistant message by the error."""
        chat = ChatState()
        chat.append_user("Recherche zu Z")
        chat.append_assistant("🔍 *Suche läuft...*")
        # the pipeline crashes
        chat.replace_last_assistant("❌ Fehler: Connection timeout")
        self.assertEqual(chat.last_assistant_content,
                         "❌ Fehler: Connection timeout")

    def test_retry_pattern(self):
        """The user corrects the input — discard the last answer."""
        chat = ChatState([
            {"role": "user", "content": "alte Frage"},
            {"role": "assistant", "content": "alte Antwort"},
        ])
        # discard: assistant answer gone
        chat.drop_last()
        self.assertEqual(chat.last_role, "user")
        # new answer
        chat.append_assistant("neue Antwort")
        self.assertEqual(chat.last_assistant_content, "neue Antwort")

    def test_safe_replace_when_chat_empty(self):
        """Edge case: replace_last_assistant on a completely empty chat.

        This is the typical trigger of an IndexError:
            chatbot[-1] = {...}  # IndexError if chatbot is empty
        ChatState catches it.
        """
        chat = ChatState()
        # must not crash!
        chat.replace_last_assistant("erste Nachricht")
        self.assertEqual(len(chat), 1)


if __name__ == "__main__":
    unittest.main()
