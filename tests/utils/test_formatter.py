import json
import pytest
from utils.formatter import (
    remove_comments,
    parse_json_from_text,
    save_as_json,
    save_as_jsonl,
    read_jsonl,
)


class TestRemoveComments:
    def test_line_comments_removed(self):
        text = '{"key": "value"} // inline comment'
        assert "inline comment" not in remove_comments(text)
        assert '"key"' in remove_comments(text)

    def test_block_comments_removed(self):
        text = '{"key": /* block */ "value"}'
        result = remove_comments(text)
        assert "block" not in result
        assert '"key"' in result

    def test_no_comments_unchanged(self):
        text = '{"key": "value"}'
        assert remove_comments(text) == text

    def test_multiline_block_comment(self):
        text = '{"key": /* multi\nline\ncomment */ "value"}'
        result = remove_comments(text)
        assert "multi" not in result
        assert "line" not in result


class TestParseJsonFromText:
    def test_plain_json(self):
        text = '{"name": "Alice", "age": 30}'
        result = parse_json_from_text(text)
        assert result == {"name": "Alice", "age": 30}

    def test_markdown_json_block(self):
        text = '```json\n{"name": "Bob"}\n```'
        result = parse_json_from_text(text)
        assert result == {"name": "Bob"}

    def test_markdown_block_no_language_tag(self):
        text = '```\n{"name": "Carol"}\n```'
        result = parse_json_from_text(text)
        assert result == {"name": "Carol"}

    def test_json_with_line_comment(self):
        text = '{"key": "value" // this is a comment\n}'
        result = parse_json_from_text(text)
        assert result.get("key") == "value"

    def test_invalid_text_returns_empty(self):
        result = parse_json_from_text("not json at all")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = parse_json_from_text("")
        assert result == {}


class TestSaveAsJson:
    def test_save_and_reload(self, tmp_path):
        data = {"field": "value", "number": 42}
        path = tmp_path / "output.json"
        save_as_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_unicode_preserved(self, tmp_path):
        data = {"text": "中文内容"}
        path = tmp_path / "unicode.json"
        save_as_json(data, path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["text"] == "中文内容"


class TestSaveAsJsonl:
    def test_save_and_reload(self, tmp_path):
        data = {"key": "val"}
        path = tmp_path / "output.jsonl"
        save_as_jsonl(data, path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == data


class TestReadJsonl:
    def _write_jsonl(self, path, records):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_read_sync_returns_list(self, tmp_path):
        path = tmp_path / "data.jsonl"
        records = [{"a": 1}, {"b": 2}]
        self._write_jsonl(path, records)
        result = read_jsonl(str(path), sync=True)
        assert result == records

    def test_read_generator(self, tmp_path):
        path = tmp_path / "data.jsonl"
        records = [{"x": 10}, {"y": 20}]
        self._write_jsonl(path, records)
        gen = read_jsonl(str(path), sync=False)
        result = list(gen)
        assert result == records

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n')
        result = read_jsonl(str(path), sync=True)
        assert len(result) == 2
