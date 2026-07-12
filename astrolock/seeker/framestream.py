"""
Named frame streams, v3: binary in-section records -- the OS can't stall a frame commit.

A STREAM ('guide', 'main_debug', ...) is a chain of SEGMENTS announced by a HEAD file
(<name>.stream.jsonl, fixed name, single writer, appended only at ROLL cadence -- the sole
disk artifact in the live path, and the discovery bootstrap):

    {"event":"segment", "shm":"<guid>"}            (or "data":"<file>.dat" for the file store)
    {"event":"ended"}

A segment is ONE section (or file) that carries everything -- v2's sidecar records moved
inside, as fixed-size binary structs (we own both ends; JSON was only convenience, and the
filesystem's write-throttle gates were stalling per-frame commits during drive breathers):

    [ header struct | JSON blob (cold metadata: geometry, settings, extras schema) ]
    [ FrameInfo[cap]: fixed binary records -- t_mono_ns, t_utc_ns, off, len, flags, src_index,
                      plus optional per-stream EXTRAS (schema declared in the header JSON) ]
    [ frame heap, 4K-aligned offsets ]

Commit = memcpy frame, fill FrameInfo[n], store header.frame_count = n+1. That single aligned
u64 store is the system's entire synchronization story (x86/ARM64 store order; a fenced build
can come later). frame_count and flags live at fixed offsets so readers poll them with pure
memory reads -- no syscalls, so even the GUI's waker probes for new frames without touching
the filesystem. Segment finalize/stream-end are header FLAG bits (+ the head's ended event).
Sections are committed lazily by the OS (untouched pages cost no RAM), sized for cap frames of
worst case -- which stays well-defined even for future variable-size frames (mjpeg bounds at
~raw+slack); the file store writes the same bytes to disk (tests/offline), patching
frame_count per commit like SerWriter used to.

Consumers are fire-and-forget: follow the head, attach segments eagerly (holding a section IS
the write-behind buffer), read records straight out of the region, finish at ended. Every file
and every section has exactly one writer.
"""

import json
import os
import struct
import uuid
from multiprocessing import shared_memory

import numpy as np

from astrolock.seeker import ser as ser_mod          # ColorId convention only
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer

_ALIGN = 4096
MAGIC = b'ALFSSEG3'
# magic 8s | version I | rec_off I | rec_size I | cap I | json_len I | pad I | count Q | flags Q
_HDR = struct.Struct('<8sIIIIIIQQ')                  # pad I keeps count/flags 8-byte ALIGNED
assert (_HDR.size - 16) % 8 == 0                     # the publish store must be a single aligned write
_COUNT_OFF = _HDR.size - 16                          # byte offset of the frame_count publish word
_FLAGS_OFF = _HDR.size - 8
FLAG_FINALIZED = 1                                   # no more frames in THIS segment
FLAG_STREAM_ENDED = 2                                # ...and the whole stream ended cleanly here

# The base per-frame record; streams may append EXTRAS (schema in the header JSON).
_REC_BASE = struct.Struct('<qqQIIq')                 # t_mono_ns, t_utc_ns, off, len, flags, src_index
_REC_FIELDS = ('t_mono_ns', 't_utc_ns', 'off', 'len', 'flags', 'src_index')


def _align(n):
    return (n + _ALIGN - 1) & ~(_ALIGN - 1)


def head_path(session_dir, name):
    return os.path.join(str(session_dir), f'{name}.stream.jsonl')


def _dtype_for(pixel_depth):
    return np.dtype('u1') if pixel_depth <= 8 else np.dtype('<u2')


def _layout(width, height, pixel_depth, cap, extras_fmt, json_len):
    frame_bytes = int(width) * int(height) * _dtype_for(pixel_depth).itemsize
    rec_size = _REC_BASE.size + (struct.calcsize(extras_fmt) if extras_fmt else 0)
    rec_off = _align(_HDR.size + json_len)
    heap_off = _align(rec_off + rec_size * cap)
    total = heap_off + _align(frame_bytes) * cap
    return frame_bytes, rec_size, rec_off, heap_off, total


# --------------------------------- write side ---------------------------------------------

class FrameStream:
    """Producer side: head file + one open segment at a time, rolled by the caller (or when
    full). Two-phase commit preserves the cam's contract: write_pixels(), any between-work
    (exposure wait), then commit(...) -- one shm store, untouchable by the filesystem."""

    def __init__(self, out_dir, name, extras=None):
        """``extras``: optional (fmt, names) -- e.g. ('<6f', ['peak','strehl',...]) -- appended
        to every frame record; schema is published in each segment's header JSON."""
        self.out_dir, self.name = str(out_dir), str(name)
        self._head = JsonlWriter(head_path(out_dir, name))
        self.extras_fmt, self.extras_names = (extras or ('', []))
        self._x = struct.Struct(self.extras_fmt) if self.extras_fmt else None
        self._seg = None                      # dict for the open segment
        self._retired_shm = None              # previous section, bridged one roll
        self._staged = None                   # (off, len) awaiting commit
        self.frame_count = 0

    def open_segment(self, seg_ts, width, height, color_id=ser_mod.ColorId.MONO,
                     pixel_depth=16, shm=True, cap=64, meta=None):
        """Roll to a fresh segment. ``meta``: cold per-segment metadata (bin/roi/settings/...)
        published in the header JSON alongside geometry and the extras schema."""
        blob = json.dumps({'name': self.name, 'seg_ts': str(seg_ts), 'width': int(width),
                           'height': int(height), 'pixel_depth': int(pixel_depth),
                           'color_id': int(color_id), 'extras_fmt': self.extras_fmt,
                           'extras': list(self.extras_names), **(meta or {})},
                          separators=(',', ':')).encode('utf-8')
        fb, rec_size, rec_off, heap_off, total = _layout(width, height, pixel_depth, cap,
                                                         self.extras_fmt, len(blob))
        hdr = _HDR.pack(MAGIC, 3, rec_off, rec_size, cap, len(blob), 0, 0, 0) + blob
        self._roll_out()
        seg = {'width': width, 'height': height, 'pixel_depth': pixel_depth, 'cap': cap,
               'rec_size': rec_size, 'rec_off': rec_off, 'heap_off': heap_off,
               'frame_bytes': fb, 'off': heap_off}
        if shm:
            section = shared_memory.SharedMemory(create=True, size=total, name=uuid.uuid4().hex)
            seg.update(kind='shm', shm=section, buf=section.buf)
            seg['buf'][:len(hdr)] = hdr
            self._head.append({'event': 'segment', 'shm': section.name})
        else:
            path = os.path.join(self.out_dir, f'{seg_ts}_{self.name}.dat')
            f = open(path, 'wb')
            f.write(hdr)
            f.flush()
            seg.update(kind='file', file=f, path=path)
            self._head.append({'event': 'segment', 'data': os.path.basename(path)})
        self._seg = seg
        self.frame_count = 0
        self._staged = None

    @property
    def full(self):
        s = self._seg
        return s is not None and self.frame_count >= s['cap']

    def write_pixels(self, frame):
        """Phase 1: place the pixels in the heap at the next aligned offset."""
        s = self._seg
        arr = np.ascontiguousarray(frame, dtype=_dtype_for(s['pixel_depth']))
        if self.frame_count >= s['cap']:
            raise ValueError(f"segment full ({s['cap']} frames); roll first")
        off, n = s['off'], arr.nbytes
        if s['kind'] == 'shm':
            s['buf'][off:off + n] = arr.tobytes()
        else:
            s['file'].seek(off)
            s['file'].write(arr.tobytes())
        self._staged = (off, n)
        s['off'] = _align(off + n)

    def commit(self, t_mono_ns=0, t_utc_ns=0, src_index=-1, flags=0, extras=()):
        """Phase 2: the frame record + the frame_count publish -- pure memory stores (shm)."""
        s = self._seg
        off, n = self._staged
        self._staged = None
        rec = _REC_BASE.pack(int(t_mono_ns), int(t_utc_ns), off, n, int(flags), int(src_index))
        if self._x is not None:
            rec += self._x.pack(*extras)
        pos = s['rec_off'] + self.frame_count * s['rec_size']
        new_count = struct.pack('<Q', self.frame_count + 1)
        if s['kind'] == 'shm':
            s['buf'][pos:pos + len(rec)] = rec
            s['buf'][_COUNT_OFF:_COUNT_OFF + 8] = new_count   # THE commit: a single u64 store
        else:
            f = s['file']
            f.seek(pos)
            f.write(rec)
            f.seek(_COUNT_OFF)
            f.write(new_count)
            f.flush()
        self.frame_count += 1

    def write(self, frame, **kw):
        self.write_pixels(frame)
        self.commit(**kw)

    def _set_flags(self, flags):
        s = self._seg
        raw = struct.pack('<Q', flags)
        if s['kind'] == 'shm':
            s['buf'][_FLAGS_OFF:_FLAGS_OFF + 8] = raw
        else:
            s['file'].seek(_FLAGS_OFF)
            s['file'].write(raw)
            s['file'].flush()

    def _roll_out(self, stream_ended=False):
        """Finalize the open segment (header flag). A rolled shm section is retained until the
        roll after next, bridging the reader-attach race."""
        if self._retired_shm is not None:
            self._retired_shm.close()
            self._retired_shm = None
        s, self._seg = self._seg, None
        if s is not None:
            self._seg = s                                    # _set_flags needs it briefly
            self._set_flags(FLAG_FINALIZED | (FLAG_STREAM_ENDED if stream_ended else 0))
            self._seg = None
            if s['kind'] == 'shm':
                s['buf'] = None
                self._retired_shm = s['shm']
            else:
                s['file'].close()

    def close(self):
        """End the stream cleanly: STREAM_ENDED flag in the segment + ended event in the head."""
        if self._head is None:
            return
        self._roll_out(stream_ended=True)
        if self._retired_shm is not None:
            self._retired_shm.close()
            self._retired_shm = None
        self._head.append({'event': 'ended'})
        self._head.close()
        self._head = None


# --------------------------------- read side ----------------------------------------------

class SegmentReader:
    """One segment: attached section (or opened .dat). committed()/finalized()/stream_ended()
    are pure memory reads of the header (thread-safe to probe -- the GUI waker does). read(i)
    returns the frame, record(i) the decoded record dict (base fields + declared extras)."""

    def __init__(self, session_dir, ref):
        self.session_dir = str(session_dir)
        self.ref = dict(ref)                 # {'shm': name} or {'data': basename}
        self._shm = None
        self._buf = None
        self._file = None
        if 'shm' in ref:
            try:
                self._shm = shared_memory.SharedMemory(name=ref['shm'], create=False)
            except (FileNotFoundError, OSError) as e:
                raise ValueError(f"shm segment {ref['shm']} gone (writer+readers exited)") from e
            self._buf = self._shm.buf
            self.ident = ref['shm']
        else:
            # File store: plain seek/read (a growing file outruns any fixed-size mapping).
            path = os.path.join(self.session_dir, ref['data'])
            self._file = open(path, 'rb')
            self.ident = ref['data']
        raw = self._get(0, _HDR.size)
        if len(raw) < _HDR.size or raw[:8] != MAGIC:
            self.close()
            raise ValueError(f"{self.ident}: bad/short segment header")
        (_, _, self._rec_off, self._rec_size, self.cap,
         json_len, _, _, _) = _HDR.unpack(raw)
        self.meta = json.loads(self._get(_HDR.size, json_len).decode('utf-8'))
        self._dtype = _dtype_for(self.meta['pixel_depth'])
        self._shape = (self.meta['height'], self.meta['width'])
        self._x = struct.Struct(self.meta['extras_fmt']) if self.meta.get('extras_fmt') else None
        self._xnames = self.meta.get('extras') or []

    def _get(self, off, n):
        if self._buf is not None:
            return bytes(self._buf[off:off + n])
        self._file.seek(off)
        return self._file.read(n)

    @property
    def header(self):
        """SerHeader-shaped view of the metadata."""
        import types
        m = self.meta
        return types.SimpleNamespace(image_width=m['width'], image_height=m['height'],
                                     color_id=m['color_id'],
                                     pixel_depth_per_plane=m['pixel_depth'])

    def committed(self):
        return struct.unpack('<Q', self._get(_COUNT_OFF, 8))[0]

    def _flags(self):
        return struct.unpack('<Q', self._get(_FLAGS_OFF, 8))[0]

    def finalized(self):
        return bool(self._flags() & FLAG_FINALIZED)

    def stream_ended(self):
        return bool(self._flags() & FLAG_STREAM_ENDED)

    def record(self, i):
        if i < 0 or i >= self.committed():
            raise IndexError(f"record {i} not committed (have {self.committed()})")
        pos = self._rec_off + i * self._rec_size
        raw = self._get(pos, self._rec_size)
        rec = dict(zip(_REC_FIELDS, _REC_BASE.unpack(raw[:_REC_BASE.size])))
        if self._x is not None:
            rec.update(zip(self._xnames, self._x.unpack(raw[_REC_BASE.size:
                                                            _REC_BASE.size + self._x.size])))
        return rec

    def read(self, i):
        rec = self.record(i)
        arr = np.frombuffer(self._get(rec['off'], rec['len']), dtype=self._dtype)
        arr = arr.reshape(self._shape).copy()                # our own buffer, not the shared region
        arr.setflags(write=False)
        return arr

    def close(self):
        self._buf = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._file is not None:
            self._file.close()
            self._file = None


class StreamFollower:
    """Consumer of a whole stream: tails the head, attaches segments eagerly (holding a section
    IS the write-behind buffer), reads records straight from the regions. Skip-to-newest
    consumers (GUI/backend) let old segments drop; sequential consumers (recorder, offline
    detect) pass keep_all=True and release() segments themselves. ended() goes True at the
    producer's clean stop."""

    def __init__(self, session_dir, name, keep_all=False):
        self.session_dir, self.name = str(session_dir), str(name)
        self.keep_all = keep_all
        self._head = None
        self._pending = []
        self.segs = []
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
                self._pending.append(ev)
            elif ev.get('event') == 'ended':
                self._ended = True
        still = []
        for ev in self._pending:
            try:
                self.segs.append(SegmentReader(self.session_dir, ev))
            except ValueError as e:
                if 'gone' in str(e):
                    self.lost += 1            # section died before we attached (count unknown)
                elif 'data' in ev and not os.path.exists(os.path.join(self.session_dir, ev['data'])):
                    pass                      # file already garbage-collected -- old history
                else:
                    still.append(ev)          # header not written yet -- retry next poll
        self._pending = still
        if not self.keep_all:                 # newest-frame consumers don't buffer history
            while len(self.segs) > 1 and self.segs[-1].committed() > 0:
                self.segs.pop(0).close()
            while len(self.segs) > 2:
                self.segs.pop(0).close()

    def latest(self):
        """(SegmentReader, index) of the newest committed frame, or None."""
        for s in reversed(self.segs):
            n = s.committed()
            if n > 0:
                return s, n - 1
        return None

    def ended(self):
        return self._ended

    @property
    def frames_path(self):
        """v3: frames commit in RAM; the head file is the stream's only on-disk artifact."""
        return None

    def release(self, seg):
        seg.close()
        self.segs.remove(seg)

    def close(self):
        for s in self.segs:
            s.close()
        self.segs = []
        if self._head is not None:
            self._head.close()
            self._head = None
