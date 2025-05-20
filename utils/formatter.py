# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import re
import logging
from typing import List, Dict, Union, Iterator

logger = logging.getLogger(__name__)


def parse_json_from_text(text: str) -> dict:
    """Extract JSON from text that may contain markdown code blocks."""
    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from markdown block")
    
    # If no markdown block found, try to parse the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse text as JSON")
        return {}

def save_as_json(data, path):
    """Save data as JSON to the specified path."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_as_jsonl(data, path):
    import json
    with open(path, 'w') as f:
        f.write(json.dumps(data) + '\n')

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
        with open(path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        # Generator version - yield one line at a time
        def line_generator():
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        yield json.loads(line)
        return line_generator()

def save_as_tsv(data, path):
    with open(path, 'w') as f:
        for item in data.get("ocr_texts", []):
            f.write("\t".join(map(str, item["bbox"])) + "\t" + item["text"] + "\n")