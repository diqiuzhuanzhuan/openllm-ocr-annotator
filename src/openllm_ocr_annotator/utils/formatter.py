# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import json
import re
import logging
from typing import List, Dict, Union, Iterator

logger = logging.getLogger(__name__)


def remove_comments(json_str: str) -> str:
    """Remove C-style comments from JSON string.

    Args:
        json_str: JSON string that may contain comments

    Returns:
        Clean JSON string with comments removed
    """
    # Remove // comments
    json_str = re.sub(r"//.*$", "", json_str, flags=re.MULTILINE)
    # Remove /* */ comments
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
    return json_str


def parse_json_from_text(text: str) -> dict:
    """Extract JSON from text that may contain markdown code blocks and comments.

    Args:
        text: Text containing JSON, possibly with markdown and comments

    Returns:
        Parsed JSON as dictionary
    """
    # Try to find JSON in markdown code block
    json_match = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL)
    if json_match:
        try:
            clean_json = remove_comments(json_match.group(1))
            return json.loads(clean_json)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from markdown block: {text}")

    # If no markdown block found, try to parse the entire text as JSON
    try:
        clean_json = remove_comments(text)
        return json.loads(clean_json)
    except json.JSONDecodeError:
        pass

    # Some reasoning models prepend text (for example <think>...</think>) before
    # the JSON payload. Try decoding from each object start until one succeeds.
    decoder = json.JSONDecoder()
    clean_text = remove_comments(text)
    for index, char in enumerate(clean_text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(clean_text[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    logger.warning(f"Failed to parse text as JSON: {text}")
    return {}


def save_as_json(data, path):
    """Save data as JSON to the specified path."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_as_jsonl(data, path):
    import json

    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")


def read_jsonl(path: str, sync: bool = True) -> Union[List[Dict], Iterator[Dict]]:
    """Read a JSONL file either all at once or line by line.

    Args:
        path: Path to the JSONL file
        sync: If True, read all lines at once and return a list
              If False, yield one line at a time

    Returns:
        If sync=True: List of dictionaries, each representing a JSON object
        If sync=False: Iterator yielding one dictionary at a time
    """
    if sync:
        # Read all lines at once
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        # Generator version - yield one line at a time
        def line_generator():
            with open(path, "r") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        yield json.loads(line)

        return line_generator()


def save_as_tsv(data, path):
    with open(path, "w") as f:
        for item in data.get("ocr_texts", []):
            f.write("\t".join(map(str, item["bbox"])) + "\t" + item["text"] + "\n")
