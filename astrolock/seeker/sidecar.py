"""
JSON-lines sidecar I/O for AstroLock Seeker.

A sidecar is an append-only stream of JSON records, one per line. The presence of a
*complete* line (terminated by a newline) is the commit point; a trailing partial line is
ignored until finished. Writers flush after each record so other processes see it
immediately via the page cache.
"""

import json


class JsonlWriter:
    """Append-only JSON-lines writer; flushes after every record."""

    def __init__(self, path):
        self.path = str(path)
        self._file = open(self.path, 'a', encoding='utf-8', newline='\n')
        self.count = 0

    def append(self, record):
        self._file.write(json.dumps(record, separators=(',', ':')))
        self._file.write('\n')
        self._file.flush()
        self.count += 1

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_complete_lines(path):
    """
    Return a list of parsed records from a (possibly growing) jsonl file, ignoring a
    trailing partial line. Returns [] if the file doesn't exist yet.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        return []
    records = []
    # Only lines followed by a newline are committed; a final line without '\n' is partial.
    end = data.rfind('\n')
    if end < 0:
        return []
    for line in data[:end].split('\n'):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def count_complete_lines(path):
    """Number of committed (newline-terminated) records, cheaply."""
    try:
        with open(path, 'rb') as f:
            return f.read().count(b'\n')
    except FileNotFoundError:
        return 0


class JsonlTailer:
    """
    Incrementally follow an append-only jsonl file. poll() returns the records completed
    since the last call (buffering any trailing partial line). Cheap for live reading -- no
    re-parsing the whole file each tick.
    """

    def __init__(self, path):
        self.path = str(path)
        self._file = None
        self._partial = ''

    def poll(self):
        out = []
        if self._file is None:
            try:
                self._file = open(self.path, 'r', encoding='utf-8')
            except FileNotFoundError:
                return out
        chunk = self._file.read()
        if chunk:
            self._partial += chunk
            if '\n' in self._partial:
                # Split once (O(n)), not by peeling one line at a time (which recopies the whole
                # remaining buffer each step -> O(n^2), seconds when ~15k lines arrive in one read).
                *lines, self._partial = self._partial.split('\n')   # last piece = incomplete tail
                for line in lines:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
