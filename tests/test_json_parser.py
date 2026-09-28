"""
Tests for `src.llm.json_parser`.

Acceptance:
  - a mock LLM that delivers JSON in a code block is parsed cleanly
  - direct, Markdown block, curly braces, square brackets and truncated
    JSON are all recognised
"""

import unittest

from src.llm.json_parser import coerce_float, parse_llm_json


class TestParseLLMJson(unittest.TestCase):

    def test_direct_json(self):
        text = '{"foo": "bar", "n": 42}'
        result = parse_llm_json(text)
        self.assertEqual(result, {"foo": "bar", "n": 42})

    def test_json_array_direct(self):
        text = '[{"a": 1}, {"b": 2}]'
        result = parse_llm_json(text)
        self.assertEqual(result, [{"a": 1}, {"b": 2}])

    def test_markdown_codeblock(self):
        text = '''Hier ist meine Antwort:
```json
{"anchor_type": "none", "confidence": 0.94}
```
'''
        result = parse_llm_json(text)
        self.assertEqual(result["anchor_type"], "none")
        self.assertEqual(result["confidence"], 0.94)

    def test_markdown_codeblock_no_lang(self):
        text = '```\n{"x": 1}\n```'
        result = parse_llm_json(text)
        self.assertEqual(result, {"x": 1})

    def test_with_prefix_text(self):
        text = 'Sicher, hier ist das JSON: {"y": 2, "z": [1, 2]}'
        result = parse_llm_json(text)
        self.assertEqual(result, {"y": 2, "z": [1, 2]})

    def test_with_think_block(self):
        text = '<think>Ich überlege...</think>\n{"answer": "ok"}'
        result = parse_llm_json(text)
        self.assertEqual(result, {"answer": "ok"})

    def test_unclosed_think_block(self):
        # With a max_tokens cutoff during reasoning the closing tag is missing.
        # The result is <think> without JSON after it — the parser must return
        # empty, not crash.
        text = '<think>Ich überlege noch und denke über'
        result = parse_llm_json(text, default={})
        self.assertEqual(result, {})

    def test_nested_braces(self):
        text = '{"outer": {"inner": {"deep": "value"}}, "x": 1}'
        result = parse_llm_json(text)
        self.assertEqual(result["outer"]["inner"]["deep"], "value")

    def test_braces_in_strings_dont_confuse_parser(self):
        # A `}` inside a string must not be taken as the end of the object.
        text = '{"text": "Hier ist ein } in einem String", "n": 1}'
        result = parse_llm_json(text)
        self.assertEqual(result["text"], "Hier ist ein } in einem String")
        self.assertEqual(result["n"], 1)

    def test_truncated_object_repaired(self):
        # Cut off at max_tokens — the parser should rescue the complete
        # fields.
        text = '{"a": 1, "b": "halbe Antwo'
        result = parse_llm_json(text, default={})
        # At least the complete first field is in it
        self.assertEqual(result.get("a"), 1)

    def test_empty_string(self):
        self.assertEqual(parse_llm_json("", default={}), {})
        self.assertEqual(parse_llm_json("   \n  ", default={}), {})

    def test_no_json_at_all(self):
        text = "Tut mir leid, ich kann das nicht."
        result = parse_llm_json(text, default={"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_expected_keys_validation_passes(self):
        text = '{"anchor_type": "person", "confidence": 0.9, "extra": "x"}'
        result = parse_llm_json(
            text, expected_keys=["anchor_type", "confidence"], default={}
        )
        self.assertEqual(result["anchor_type"], "person")

    def test_expected_keys_validation_fails(self):
        # `confidence` missing — the parser must return the default
        text = '{"anchor_type": "person"}'
        result = parse_llm_json(
            text, expected_keys=["anchor_type", "confidence"], default={}
        )
        self.assertEqual(result, {})

    def test_array_after_text(self):
        text = "Hier sind die Items:\n[{\"i\": 1}, {\"i\": 2}]"
        result = parse_llm_json(text)
        self.assertEqual(result, [{"i": 1}, {"i": 2}])


class TestCoerceFloat(unittest.TestCase):

    def test_float_passthrough(self):
        self.assertEqual(coerce_float(0.85), 0.85)

    def test_int_to_float(self):
        self.assertEqual(coerce_float(1), 1.0)

    def test_string_with_dot(self):
        self.assertEqual(coerce_float("0.7"), 0.7)

    def test_string_with_comma(self):
        # German spelling — tolerated
        self.assertEqual(coerce_float("0,7"), 0.7)

    def test_percent_string(self):
        self.assertEqual(coerce_float("85%"), 0.85)

    def test_percent_too_big_normalized(self):
        # The LLM answers "0.85" as "85" — recognised heuristically
        result = coerce_float("85")
        self.assertEqual(result, 0.85)

    def test_clamp_below(self):
        self.assertEqual(coerce_float(-0.5), 0.0)

    def test_clamp_above(self):
        self.assertEqual(coerce_float(1.5), 1.0)

    def test_invalid_returns_default(self):
        self.assertEqual(coerce_float("not a number", default=0.42), 0.42)

    def test_none_returns_default(self):
        self.assertEqual(coerce_float(None, default=0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
