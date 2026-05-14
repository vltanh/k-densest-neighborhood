"""Lightweight record I/O helpers shared by classification scripts."""

import json
import os
import sys
from typing import List


def load_ndjson_records(path: str) -> List[dict]:
    """Read an NDJSON file of records and report malformed lines to stderr."""
    if not os.path.exists(path):
        return []
    out: List[dict] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped > 0:
        print(
            f"[load_ndjson_records] {path}: {skipped} malformed line(s) skipped",
            file=sys.stderr,
        )
    return out
