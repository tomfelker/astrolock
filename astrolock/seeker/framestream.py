"""
Named frame streams, v2: chained sidecars, head files, and pure frame-heap stores.

A STREAM ('guide', 'main_debug', ...) is a chain of SEGMENTS. Discovery involves no globbing
and no stat-polling of sentinel files:

  <name>.stream.jsonl            the HEAD file (fixed name, single writer = the producer):
                                   {"event":"segment","sidecar":"<ts>_<name>.frames.jsonl"}
                                   {"event":"ended"}
  <ts>_<name>.frames.jsonl       a segment's sidecar -- the commit spine and the ONLY protocol:
                                   {"type":"start", store/geometry metadata}          (line 0)
                                   {"off":..,"len":.., ...}    one per frame, in order
                                   {"type":"next","sidecar":"..."}  on roll, OR
                                   {"type":"ended"}                 on clean stream end

The pixel store is a pure frame HEAP -- a shared-memory section (named by a random GUID carried
in the start record; nothing lands on disk) or a plain .dat file -- with frames written at
4K-aligned offsets and addressed ONLY by the sidecar's off/len records. There is no format
inside the store, no counters, no locks: the writer fills the pixels, THEN write()s the sidecar
line, and that syscall is the commit point and the memory barrier. Committed frames are
immutable. Variable-size frames (mjpeg, compressed archives) need nothing new.

Consumers are fire-and-forget: open the head, follow the chain, stop at 'ended' (a clean
shutdown is an event in the data, not a sentinel file); a chain that never ends but goes stale
means the producer died. Sections die with their last handle (Windows named sections are
kernel-refcounted, session-local, same-user), so a reader holding a segment IS the write-behind
buffer, and cleanup is garbage collection of small sidecar files only.

Why every file has exactly ONE writer (head = producer, sidecar = producer, heap = producer):
multi-process appends to a shared index would be the one racy spot in an otherwise race-free
design. See docs/notes.md 2026-07-10/11 for the SSD saga that motivated all of this.
"""

import json
import os
import uuid
from multiprocessing import shared_memory

import numpy as np

from astrolock.seeker import ser as ser_mod          # ColorId + container/dtype conventions only
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer

_ALIGN = 4096                                        # frame offsets are 4K-aligned (for luck)


def _align(n):
    return (n + _ALIGN - 1) & ~(_ALIGN - 1)


def head_path(session_dir, name):
    return os.path.join(str(session_dir), f'{name}.stream.jsonl')


def _dtype_for(pixel_depth):
    return np.dtype('u1') if pixel_depth <= 8 else np.dtype('<u2')


# --------------------------------- write side ---------------------------------------------

class FrameStream:
    """Producer side of a named stream: head file + one open segment at a time, rolled by the
    caller (or when full). Two-phase commit preserves the cam's contract: write_pixels(), any
    between-work (exposure wait), then commit(**record) -- the sidecar line IS the commit."""

    def __init__(self, out_dir, name):
        self.out_dir, self.name = str(out_dir), str(name)
        self._head = JsonlWriter(head_path(out_dir, name))
        self._sidecar = None
        self._store = None                    # dict: kind/shm/file handles for the open segment
        self._retired_shm = None              # previous segment's section, bridged one roll
        self._staged = None                   # (off, len) from write_pixels awaiting commit
        self._off = 0
        self.frame_count = 0                  # frames committed in the CURRENT segment
        self.frames_path = None
        self._meta = None

    def open_segment(self, seg_ts, width, height, color_id=ser_mod.ColorId.MONO,
                     pixel_depth=16, shm=True, cap=64):
        """Roll to a fresh segment '<seg_ts>_<name>': create the new store + sidecar first, then
        chain the old sidecar to it ('next') -- a reader only ever sees a link to something that
        already exists."""
        stem = os.path.join(self.out_dir, f'{seg_ts}_{self.name}')
        new_frames = stem + '.frames.jsonl'
        frame_bytes = int(width) * int(height) * _dtype_for(pixel_depth).itemsize
        cap_bytes = _align(frame_bytes) * int(cap)
        meta = {'type': 'start', 'width': int(width), 'height': int(height),
                'pixel_depth': int(pixel_depth), 'color_id': int(color_id)}
        if shm:
            section = shared_memory.SharedMemory(create=True, size=cap_bytes,
                                                 name=uuid.uuid4().hex)
            store = {'kind': 'shm', 'shm': section, 'buf': section.buf, 'cap': cap_bytes}
            meta.update(store='shm', shm=section.name)
        else:
            f = open(stem + '.dat', 'wb')
            store = {'kind': 'file', 'file': f, 'cap': None}
            meta.update(store='file', data=os.path.basename(stem) + '.dat')
        sc = JsonlWriter(new_frames)
        sc.append(meta)
        self._chain_out(next_sidecar=os.path.basename(new_frames))
        self._sidecar, self._store, self._meta = sc, store, meta
        self._off = 0
        self.frame_count = 0
        self._staged = None
        self.frames_path = new_frames
        self._head.append({'event': 'segment', 'sidecar': os.path.basename(new_frames)})

    @property
    def full(self):
        """True when the next fixed-size frame wouldn't fit (shm segments are committed RAM)."""
        st = self._store
        if st is None or st['cap'] is None or self._meta is None:
            return False
        fb = _align(self._meta['width'] * self._meta['height']
                    * _dtype_for(self._meta['pixel_depth']).itemsize)
        return self._off + fb > st['cap']

    def write_pixels(self, frame):
        """Phase 1: place the pixels in the heap at the next aligned offset."""
        arr = np.ascontiguousarray(frame, dtype=_dtype_for(self._meta['pixel_depth']))
        off, n = self._off, arr.nbytes
        st = self._store
        if st['kind'] == 'shm':
            if off + n > st['cap']:
                raise ValueError(f"segment full ({st['cap']} bytes); roll first")
            st['buf'][off:off + n] = arr.tobytes()
        else:
            f = st['file']
            f.seek(off)
            f.write(arr.tobytes())
            f.flush()
        self._staged = (off, n)
        self._off = _align(off + n)

    def commit(self, **record):
        """Phase 2: the sidecar line -- THE commit point (and, cross-process, the memory barrier:
        readers only trust pixels a complete line points at)."""
        off, n = self._staged
        self._staged = None
        self._sidecar.append({'off': off, 'len': n, **record})
        self.frame_count += 1

    def write(self, frame, **record):
        self.write_pixels(frame)
        self.commit(**record)

    def _chain_out(self, next_sidecar=None):
        """Close the open segment, ending its sidecar with 'next' (roll) or 'ended' (stream end).
        A rolled shm section is retained until the roll after next, so a reader that hasn't
        attached yet can't lose the race."""
        if self._sidecar is not None:
            self._sidecar.append({'type': 'next', 'sidecar': next_sidecar} if next_sidecar
                                 else {'type': 'ended'})
            self._sidecar.close()
            self._sidecar = None
        if self._retired_shm is not None:
            self._retired_shm.close()
            self._retired_shm = None
        st, self._store = self._store, None
        if st is not None:
            if st['kind'] == 'shm':
                st['buf'] = None
                self._retired_shm = st['shm']
            else:
                st['file'].close()

    def close(self):
        """End the stream cleanly: 'ended' in the sidecar AND the head; consumers self-finish."""
        if self._head is None:
            return
        self._chain_out(next_sidecar=None)
        if self._retired_shm is not None:
            self._retired_shm.close()
            self._retired_shm = None
        self._head.append({'event': 'ended'})
        self._head.close()
        self._head = None


# --------------------------------- read side ----------------------------------------------

class SegmentReader:
    """One segment: its tailed sidecar + attached heap. read(i) returns frame i (immutable,
    read-only). committed() grows as the producer commits; next_sidecar()/stream_ended() report
    the chain link once present."""

    def __init__(self, session_dir, frames_path):
        self.session_dir = str(session_dir)
        self.frames_path = str(frames_path)
        self._tail = JsonlTailer(self.frames_path)
        first = self._tail.poll()
        if not first or first[0].get('type') != 'start':
            self._tail.close()
            raise ValueError(f"{frames_path}: no committed start record yet")
        self.meta = first[0]
        self.recs = [r for r in first[1:] if 'off' in r]
        self._link = next((r for r in first[1:] if r.get('type') in ('next', 'ended')), None)
        self._dtype = _dtype_for(self.meta['pixel_depth'])
        self._shape = (self.meta['height'], self.meta['width'])
        self._shm = None
        self._buf = None
        if self.meta['store'] == 'shm':
            try:
                self._shm = shared_memory.SharedMemory(name=self.meta['shm'], create=False)
            except (FileNotFoundError, OSError) as e:
                self._tail.close()
                raise ValueError(f"{frames_path}: shm segment gone (writer+readers exited)") from e
            self._buf = self._shm.buf
        else:
            self._data_path = os.path.join(self.session_dir, self.meta['data'])

    @property
    def header(self):
        """SerHeader-shaped view of the metadata (image_width/height/color_id/pixel_depth...)."""
        import types
        m = self.meta
        return types.SimpleNamespace(image_width=m['width'], image_height=m['height'],
                                     color_id=m['color_id'],
                                     pixel_depth_per_plane=m['pixel_depth'])

    def poll(self):
        """Ingest newly committed records; call before reading counts/links."""
        for r in self._tail.poll():
            if 'off' in r:
                self.recs.append(r)
            elif r.get('type') in ('next', 'ended'):
                self._link = r

    def committed(self):
        return len(self.recs)

    def finalized(self):
        return self._link is not None

    def next_sidecar(self):
        return self._link['sidecar'] if self._link and self._link.get('type') == 'next' else None

    def stream_ended(self):
        return bool(self._link and self._link.get('type') == 'ended')

    def read(self, i):
        if i < 0 or i >= len(self.recs):
            raise IndexError(f"frame {i} not committed (have {len(self.recs)})")
        rec = self.recs[i]
        if self._buf is not None:
            arr = np.frombuffer(self._buf, dtype=self._dtype,
                                count=rec['len'] // self._dtype.itemsize, offset=rec['off'])
            arr = np.array(arr).reshape(self._shape)       # copy out of the shared region
        else:
            arr = np.fromfile(self._data_path, dtype=self._dtype,
                              count=rec['len'] // self._dtype.itemsize, offset=rec['off'])
            arr = arr.reshape(self._shape)
        arr.setflags(write=False)
        return arr

    def close(self):
        self._tail.close()
        if self._shm is not None:
            self._buf = None
            self._shm.close()
            self._shm = None


class StreamFollower:
    """Consumer of a whole stream: tails the head file, opens segments as they appear, follows
    the chain. Two consumption styles:
      * skip-to-newest (GUI/backend): poll(); latest() -> (segment, index) of the newest frame.
      * sequential (detect offline, recorder): poll(); segments() in order; each SegmentReader
        is drained by the caller at its own pace -- holding it holds the shm section alive.
    ended() goes True when the producer wrote its 'ended' event (clean stop)."""

    def __init__(self, session_dir, name, keep_all=False):
        self.session_dir, self.name = str(session_dir), str(name)
        self.keep_all = keep_all              # sequential consumers keep every segment until released
        self._head = None
        self._pending = []                    # sidecar basenames announced but not yet openable
        self.segs = []                        # opened SegmentReaders, oldest first
        self.lost = 0                         # frames in segments that vanished before we attached
        self._ended = False

    def poll(self):
        if self._head is None:
            hp = head_path(self.session_dir, self.name)
            if not os.path.exists(hp):
                return
            self._head = JsonlTailer(hp)
        for ev in self._head.poll():
            if ev.get('event') == 'segment':
                self._pending.append(ev['sidecar'])
            elif ev.get('event') == 'ended':
                self._ended = True
        still = []
        for base in self._pending:            # attach EAGERLY: holding the reader holds the section
            full = os.path.join(self.session_dir, base)
            try:
                self.segs.append(SegmentReader(self.session_dir, full))
            except ValueError as e:
                if 'gone' in str(e):          # vanished before we attached: count and move on
                    from astrolock.seeker import sidecar as _sc
                    self.lost += max(0, _sc.count_complete_lines(full) - 2)
                elif not os.path.exists(full):
                    pass                      # sidecar already garbage-collected -- old history, skip
                else:
                    still.append(base)        # start record not committed yet -- retry next poll
        self._pending = still
        for s in self.segs:
            s.poll()
        if not self.keep_all:                 # newest-frame consumers don't buffer history
            while len(self.segs) > 2:
                self.segs.pop(0).close()
            while len(self.segs) > 1 and self.segs[-1].committed() > 0:
                self.segs.pop(0).close()

    def latest(self):
        """(SegmentReader, index) of the newest committed frame, or None."""
        for s in reversed(self.segs):
            if s.committed() > 0:
                return s, s.committed() - 1
        return None

    def ended(self):
        return self._ended

    @property
    def frames_path(self):
        """The newest segment's sidecar (the on-disk file that grows per frame -- watch this)."""
        return self.segs[-1].frames_path if self.segs else None

    def release(self, seg):
        """Sequential consumers: done with a fully-drained segment."""
        seg.close()
        self.segs.remove(seg)

    def close(self):
        for s in self.segs:
            s.close()
        self.segs = []
        if self._head is not None:
            self._head.close()
            self._head = None
