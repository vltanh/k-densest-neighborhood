"""Lightweight record I/O helpers shared by classification scripts."""

import json
import os
import sys
from typing import Iterator, List


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


def path_touches_backup(path: str) -> bool:
    """Return True when any path component is a backup directory."""
    return any(part.endswith(".bak") for part in os.path.normpath(path).split(os.sep))


def iter_record_files(path: str) -> Iterator[str]:
    """Yield active records.ndjson files under path, pruning backup dirs."""
    if path_touches_backup(path):
        return
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.endswith(".bak")]
            if path_touches_backup(root):
                dirs[:] = []
                continue
            for fn in files:
                if fn == "records.ndjson":
                    yield os.path.join(root, fn)
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    yield path


def load_records_from_path(path: str) -> List[dict]:
    """Load records from one NDJSON file or an active directory tree."""
    records: List[dict] = []
    for record_file in iter_record_files(path):
        records.extend(load_ndjson_records(record_file))
    return records
