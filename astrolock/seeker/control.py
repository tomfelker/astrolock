"""
Live command channel between the GUI and the backend (the one non-file path in seeker).

The backend runs a CommandServer (accepts newline-JSON commands into a thread-safe queue);
the control loop drains it non-blocking so a slow/dead GUI can never stall tracking. The
GUI uses a CommandClient to send. Command shapes (see astrolock_seeker.md):

    {"type": "set_rate", "az": <deg/s>, "alt": <deg/s>}   # manual slew
    {"type": "stop"}                                       # zero rates
    {"type": "estop"}                                      # zero + latch idle
"""

import json
import queue
import socket
import sys
import threading
import time


class CommandServer:
    """Threaded TCP server; commands land in a queue the control loop drains via drain()."""

    def __init__(self, host='127.0.0.1', port=0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)
        self.host, self.port = self._sock.getsockname()
        self._q = queue.Queue()
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        self._sock.settimeout(0.5)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        conn.settimeout(0.5)
        buf = ''
        while not self._stop:
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buf += data.decode('utf-8', errors='ignore')
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()
                if line:
                    try:
                        self._q.put(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        conn.close()

    def drain(self):
        """Return all commands received since the last call (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


class ControlReader:
    """
    Tail a control source -- a JSONL file or '-' for stdin -- on a background thread, parsing
    complete lines into a queue the capture loop drains each frame. A thread keeps it simple
    and portable (no non-blocking-stdin tricks); control files are tiny and page-cached.

    Each line is a JSON object of settings to merge (only the keys present change).
    """

    def __init__(self, source):
        self.source = source
        self._q = queue.Queue()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _push(self, line):
        line = line.strip()
        if line:
            try:
                self._q.put(json.loads(line))
            except json.JSONDecodeError:
                pass

    def _run(self):
        if self.source == '-':
            for line in sys.stdin:          # blocks naturally between lines
                if self._stop:
                    break
                self._push(line)
            return
        f = None
        while not self._stop:
            if f is None:
                try:
                    f = open(self.source, 'r', encoding='utf-8')
                except FileNotFoundError:
                    time.sleep(0.1)
                    continue
            line = f.readline()
            if line == '':                  # at EOF: wait for the file to grow
                time.sleep(0.05)
                continue
            self._push(line)

    def drain(self):
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._stop = True


class CommandClient:
    """Non-blocking sender. send() never blocks; drops on backpressure/disconnect."""

    def __init__(self, host, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        self._sock.setblocking(False)

    def send(self, obj):
        try:
            self._sock.sendall((json.dumps(obj) + '\n').encode('utf-8'))
        except (BlockingIOError, OSError):
            pass

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
