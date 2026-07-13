"""
framestream: one shared-memory RING per stream -- zero syscalls, zero copies in the hot loops.

The store is a single section per stream (per geometry), laid out as
    [ header + cold JSON | FrameInfo[N] record ring | N fixed-size frame slots ]
Frame ``i`` -- an ABSOLUTE index that never wraps or restarts -- lives in slot ``i % N``. Two
u64 counters in the header carry the whole protocol, each written with a single aligned store:

    C (claim):  the writer stores C = j immediately BEFORE touching slot j % N
    W (commit): the writer stores W = j + 1 after the pixels + record have landed

A reader may use any frame with ``i < W`` ZERO-COPY (a view straight into the section: SER
write(), texture upload, or a float-convert kernel reads it in place), then validates
``C < i + N`` afterwards -- if the writer claimed that slot mid-use, the read raises Lapped and
the consumer retries from ``first_available()``. The newest frame can never be lapped, so
latest-frame consumers (GUI, servo) never even see the boundary. N is the stream's whole
write-behind budget: a consumer more than N frames behind loses frames, explicitly counted.

The ring is an IMPLEMENTATION DETAIL. The public surface is absolute indices + views + Lapped;
a segmented (or any other) store could come back behind the same API with changes local to
this file. Discovery is unchanged: a per-stream head file <name>.stream.jsonl (single writer)
announces the ring section (a fresh one on a geometry change -- the only "roll" left, which is
also the only legitimate reset point for consumers' temporal state) and the clean 'ended'.
State stays in LatestSlot; nothing here touches the disk after startup.
"""

import json
import os
import struct
import uuid
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np

from astrolock.seeker import ser as ser_mod          # ColorId convention only
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer

_ALIGN = 4096
MAGIC = b'ALFSRNG1'
# magic 8s | version I | rec_off I | rec_size I | cap I | json_len I | pad I |
#   slot_bytes Q | claim Q | count Q | flags Q          (u64s 8-byte aligned)
_HDR = struct.Struct('<8sIIIIIIQQQQ')
_SLOTB_OFF = _HDR.size - 32
_CLAIM_OFF = _HDR.size - 24
_COUNT_OFF = _HDR.size - 16
_FLAGS_OFF = _HDR.size - 8
FLAG_FINALIZED = 1                                   # no more frames in THIS ring (reconfigured)
FLAG_STREAM_ENDED = 2                                # ...and the whole stream ended cleanly here

# The base per-frame record; streams may append EXTRAS (schema in the header JSON).
_REC_BASE = struct.Struct('<qqQIIq')                 # t_mono_ns, t_utc_ns, off, len, flags, src_index
_REC_FIELDS = ('t_mono_ns', 't_utc_ns', 'off', 'len', 'flags', 'src_index')


class Lapped(Exception):
    """The writer overwrote (or claimed) this frame's slot: the consumer fell > N behind, or
    the slot was reused mid-view. Retry from first_available(), counting the loss."""


def _align(n):
    return (n + _ALIGN - 1) & ~(_ALIGN - 1)


def head_path(session_dir, name):
    return os.path.join(str(session_dir), f'{name}.stream.jsonl')


def _dtype_for(pixel_depth):
    return np.uint16 if pixel_depth > 8 else np.uint8


def _u64(buf, off):
    return struct.unpack_from('<Q', buf, off)[0]


class FrameStream:
    """Producer side. configure() once per geometry (the ring is fixed-slot); then the cam's
    two-phase contract: write_pixels()/begin_frame() -> any between-work -> commit(...), where
    the commit is one aligned u64 store. No syscalls, no allocation, no unmapping per frame."""

    def __init__(self, out_dir, name, extras=None):
        """``extras``: optional (fmt, names) -- e.g. ('<8f', ['peak','strehl',...]) -- appended
        to every frame record; schema is published in the ring's header JSON."""
        self.out_dir, self.name = str(out_dir), str(name)
        self._head = JsonlWriter(head_path(out_dir, name))
        self.extras_fmt, self.extras_names = (extras or ('', []))
        self._x = struct.Struct(self.extras_fmt) if self.extras_fmt else None
        self._ring = None                     # dict for the live ring
        self._retired = None                  # previous section, bridged until the next configure
        self._staged = None                   # (off, len) awaiting commit
        self._raw = False
        self.count = 0                        # committed frames in the live ring (absolute index)

    @property
    def configured(self):
        return self._ring is not None

    def configure(self, width, height, color_id=ser_mod.ColorId.MONO, pixel_depth=16,
                  shm=True, frames=64, meta=None, raw=False):
        """(Re)create the ring for a geometry. ``frames`` = N, the stream's entire write-behind
        budget (RAM = N x slot). ``raw``: payloads are variable-size byte blobs; width x height
        x depth only sizes the worst-case slot; readers get bytes, not a reshaped image."""
        blob = json.dumps({'name': self.name, 'width': int(width), 'height': int(height),
                           'pixel_depth': int(pixel_depth), 'color_id': int(color_id),
                           'extras_fmt': self.extras_fmt, 'extras': list(self.extras_names),
                           **({'raw': True} if raw else {}), **(meta or {})},
                          separators=(',', ':')).encode('utf-8')
        cap = max(2, int(frames))
        slot = _align(int(width) * int(height) * (2 if pixel_depth > 8 else 1))
        self._raw = bool(raw)
        rec_size = _REC_BASE.size + (self._x.size if self._x else 0)
        rec_off = _align(_HDR.size + len(blob))
        heap_off = _align(rec_off + cap * rec_size)
        total = heap_off + cap * slot
        hdr = _HDR.pack(MAGIC, 1, rec_off, rec_size, cap, len(blob), 0, slot, 0, 0, 0) + blob
        self._finalize_ring()
        ring = {'cap': cap, 'slot': slot, 'rec_size': rec_size,
                'rec_off': rec_off, 'heap_off': heap_off,
                'width': int(width), 'height': int(height), 'pixel_depth': int(pixel_depth)}
        if shm:
            section = shared_memory.SharedMemory(create=True, size=total, name=uuid.uuid4().hex)
            ring.update(kind='shm', shm=section, buf=section.buf, ident=section.name)
            ring['buf'][:len(hdr)] = hdr
            self._head.append({'event': 'ring', 'shm': section.name})
        else:
            path = os.path.join(self.out_dir, f'{self.name}_{uuid.uuid4().hex[:8]}.dat')
            f = open(path, 'w+b')
            f.truncate(total)                 # sparse where the OS allows; tests/offline only
            f.write(hdr)
            f.flush()
            ring.update(kind='file', file=f, path=path, ident=os.path.basename(path))
            self._head.append({'event': 'ring', 'data': ring['ident']})
        self._ring = ring
        self.count = 0
        self._staged = None

    @property
    def ident(self):
        return self._ring['ident'] if self._ring else None

    def _store(self, off, data):
        r = self._ring
        if r['kind'] == 'shm':
            r['buf'][off:off + len(data)] = data
        else:
            r['file'].seek(off)
            r['file'].write(data)

    def _claim(self, j):
        # The claim precedes any touch of slot j % cap: a reader seeing C >= i + cap knows
        # frame i's slot is dirty. (Single aligned u64 store.)
        self._store(_CLAIM_OFF, struct.pack('<Q', j))

    def begin_frame(self):
        """Zero-copy producer path: claim the next slot and return a writable view of it.
        Fill it (or as much as the frame needs), then commit(). shm rings only."""
        r = self._ring
        self._claim(self.count)
        off = r['heap_off'] + (self.count % r['cap']) * r['slot']
        self._staged = (off, r['slot'])
        if r['kind'] != 'shm':
            raise RuntimeError('begin_frame() needs a shm ring; use write_pixels()')
        return np.frombuffer(r['buf'], dtype=np.uint8, count=r['slot'], offset=off)

    def write_pixels(self, frame):
        """Copy-in producer path (drivers hand us their own arrays anyway)."""
        r = self._ring
        arr = (np.ascontiguousarray(frame) if self._raw else
               np.ascontiguousarray(frame, dtype=_dtype_for(r['pixel_depth'])))
        n = arr.nbytes
        if n > r['slot']:
            raise ValueError(f'frame ({n}B) exceeds the ring slot ({r["slot"]}B)')
        self._claim(self.count)
        off = r['heap_off'] + (self.count % r['cap']) * r['slot']
        if r['kind'] == 'shm':
            r['buf'][off:off + n] = arr.tobytes()
        else:
            r['file'].seek(off)
            r['file'].write(arr.tobytes())
        self._staged = (off, n)

    def commit(self, t_mono_ns=0, t_utc_ns=0, src_index=-1, flags=0, extras=(), length=None):
        """Publish the staged frame: record into the record ring, then W -- one aligned store.
        ``length`` overrides the payload length (begin_frame() stages the whole slot)."""
        r = self._ring
        off, n = self._staged
        self._staged = None
        if length is not None:
            n = int(length)
        rec = _REC_BASE.pack(int(t_mono_ns), int(t_utc_ns), off, n, int(flags), int(src_index))
        if self._x is not None:
            rec += self._x.pack(*extras)
        self._store(r['rec_off'] + (self.count % r['cap']) * r['rec_size'], rec)
        self._store(_COUNT_OFF, struct.pack('<Q', self.count + 1))   # THE commit
        if r['kind'] == 'file':
            r['file'].flush()
        self.count += 1

    def write(self, frame, **kw):
        self.write_pixels(frame)
        self.commit(**kw)

    def _set_flags(self, flags):
        self._store(_FLAGS_OFF, struct.pack('<Q', flags))
        if self._ring['kind'] == 'file':
            self._ring['file'].flush()

    def _finalize_ring(self, stream_ended=False):
        """Mark the live ring finished (a reconfigure or the stream end). The section itself is
        retired but held until the next configure/close, bridging the reader-attach race; a
        FINALIZED ring is immutable, so late readers may drain it safely at leisure."""
        if self._retired is not None:
            if self._retired.get('kind') == 'shm':
                self._retired['buf'] = None
                self._retired['shm'].close()
            else:
                self._retired['file'].close()
            self._retired = None
        r, self._ring = self._ring, None
        if r is not None:
            self._ring = r                                   # _set_flags needs it briefly
            self._set_flags(FLAG_FINALIZED | (FLAG_STREAM_ENDED if stream_ended else 0))
            self._ring = None
            self._retired = r

    def close(self):
        """End the stream cleanly: STREAM_ENDED flag in the ring + ended event in the head."""
        if self._head is None:
            return
        self._finalize_ring(stream_ended=True)
        if self._retired is not None:
            if self._retired.get('kind') == 'shm':
                self._retired['buf'] = None
                self._retired['shm'].close()
            else:
                self._retired['file'].close()
            self._retired = None
        self._head.append({'event': 'ended'})
        self._head.close()
        self._head = None


# --------------------------------- read side ----------------------------------------------


class RingReader:
    """One attached ring. committed()/claim()/flags are pure memory reads (safe to probe from
    any thread -- the GUI waker does). view(i) is the zero-copy read: a context manager whose
    EXIT validates the slot wasn't reused mid-use (else Lapped). read(i)/record(i) build on it."""

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
                raise ValueError(f"ring {ref['shm']} gone (writer+readers exited)") from e
            self._buf = self._shm.buf
            self.ident = ref['shm']
        else:
            path = os.path.join(self.session_dir, ref['data'])
            # Unbuffered: the counters are live shared state; a BufferedReader can serve a
            # seek from its stale internal buffer. File store is tests/offline only.
            self._file = open(path, 'rb', buffering=0)
            self.ident = ref['data']
        raw = self._get(0, _HDR.size)
        if len(raw) < _HDR.size or raw[:8] != MAGIC:
            self.close()
            raise ValueError(f'{self.ident}: bad/short ring header')
        (_, _, self._rec_off, self._rec_size, self.cap,
         json_len, _, self.slot_bytes, _, _, _) = _HDR.unpack(raw)
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
        if self._buf is not None:
            return _u64(self._buf, _COUNT_OFF)
        return _u64(self._get(_COUNT_OFF, 8), 0)

    def claim(self):
        if self._buf is not None:
            return _u64(self._buf, _CLAIM_OFF)
        return _u64(self._get(_CLAIM_OFF, 8), 0)

    def _flags(self):
        if self._buf is not None:
            return _u64(self._buf, _FLAGS_OFF)
        return _u64(self._get(_FLAGS_OFF, 8), 0)

    def finalized(self):
        return bool(self._flags() & FLAG_FINALIZED)

    def stream_ended(self):
        return bool(self._flags() & FLAG_STREAM_ENDED)

    def first_available(self):
        """Oldest index that is still safe to read (the lap boundary)."""
        return max(0, self.claim() - self.cap + 1)

    def _check(self, i):
        if i < 0 or i >= self.committed():
            raise IndexError(f'frame {i} not committed (have {self.committed()})')
        if self.claim() >= i + self.cap:
            raise Lapped(f'frame {i} overwritten (claim {self.claim()}, cap {self.cap})')

    def record(self, i):
        """Decoded record dict for frame i (base fields + declared extras)."""
        self._check(i)
        raw = self._get(self._rec_off + (i % self.cap) * self._rec_size, self._rec_size)
        if self.claim() >= i + self.cap:                     # reused while we read it
            raise Lapped(f'record {i} overwritten mid-read')
        rec = dict(zip(_REC_FIELDS, _REC_BASE.unpack(raw[:_REC_BASE.size])))
        if self._x is not None:
            rec.update(zip(self._xnames, self._x.unpack(raw[_REC_BASE.size:
                                                            _REC_BASE.size + self._x.size])))
        return rec

    @contextmanager
    def view(self, i):
        """ZERO-COPY read of frame i: yields (record, payload_view) -- a numpy view (image) or
        memoryview (raw stream) straight into the section. Consume it (file write, texture
        upload, device copy, kernel); on exit the slot is validated and Lapped raised if the
        writer reused it mid-use, in which case DISCARD whatever was produced from the view.
        The view must NOT outlive the with-block: a lingering reference pins the mapping and
        close() will refuse with BufferError (loud by design -- copy if you must keep it)."""
        rec = self.record(i)
        if self._buf is not None:
            mv = self._buf[rec['off']:rec['off'] + rec['len']]
            if self.meta.get('raw'):
                payload = mv
            else:
                payload = np.frombuffer(mv, dtype=self._dtype).reshape(self._shape)
                payload.setflags(write=False)
        else:
            data = self._get(rec['off'], rec['len'])         # file store: the read IS the copy
            payload = data if self.meta.get('raw') else \
                np.frombuffer(data, dtype=self._dtype).reshape(self._shape)
        yield rec, payload
        # Normal exit: validate the slot survived the whole use. (On an exception in the body
        # the consumer is already bailing; no point stacking a Lapped on top.)
        if self.claim() >= i + self.cap:
            raise Lapped(f'frame {i} overwritten mid-view')

    def read(self, i):
        """Copying convenience read (tests, small/raw streams): bytes for raw, array otherwise."""
        with self.view(i) as (_rec, payload):
            out = bytes(payload) if self.meta.get('raw') else payload.copy()
        if isinstance(out, np.ndarray):
            out.setflags(write=False)
        return out

    def close(self):
        self._buf = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._file is not None:
            self._file.close()
            self._file = None


class StreamFollower:
    """Consumer of one stream: tails the head, attaches rings as they appear. The surface is
    absolute indices + latest()/drain() -- no store concepts. drain() is the sequential path:
    it walks every committed frame in order across ring changes, skipping (and counting, in
    .lost) anything the writer lapped."""

    def __init__(self, session_dir, name):
        self.session_dir, self.name = str(session_dir), str(name)
        self._head = None
        self._pending = []
        self._readers = []                    # oldest..newest attached rings (<= 2 in practice)
        self._drained = {}                    # ident -> next index for drain()
        self.lost = 0                         # frames lost to lapping / vanished rings
        self._ended = False

    def poll(self):
        if self._head is None:
            hp = head_path(self.session_dir, self.name)
            if not os.path.exists(hp):
                return
            self._head = JsonlTailer(hp)
        for ev in self._head.poll():
            if ev.get('event') in ('ring', 'segment'):
                self._pending.append(ev)
                # A ring AFTER an ended = the producer was relaunched (Connect / source switch
                # appends to the same head): 'ended' closed the PREVIOUS run, not the stream
                # forever. Without this, ended is sticky and every consumer started after a cam
                # relaunch drains the backlog and then quietly exits at its ended() check.
                self._ended = False
            elif ev.get('event') == 'ended':
                self._ended = True
        still = []
        for ev in self._pending:
            try:
                self._readers.append(RingReader(self.session_dir, ev))
            except ValueError as e:
                if 'gone' in str(e):
                    self.lost += 1            # ring died before we attached (count unknown)
                elif 'data' in ev and not os.path.exists(os.path.join(self.session_dir, ev['data'])):
                    pass                      # file already cleaned up -- old history
                else:
                    still.append(ev)          # header not written yet -- retry next poll
        self._pending = still
        # Old rings are immutable once finalized; drop any the drain cursor has fully consumed
        # (or that nobody will consume) once a newer ring exists.
        while len(self._readers) > 1:
            r = self._readers[0]
            if not r.finalized():
                break
            if self._drained.get(r.ident, 0) < r.committed():
                break
            self._drained.pop(r.ident, None)
            r.close()
            self._readers.pop(0)

    @property
    def current(self):
        """The newest attached ring (geometry/meta/ident of the live stream), or None."""
        return self._readers[-1] if self._readers else None

    @property
    def meta(self):
        r = self.current
        return r.meta if r is not None else None

    @property
    def header(self):
        r = self.current
        return r.header if r is not None else None

    @property
    def ident(self):
        r = self.current
        return r.ident if r is not None else None

    def committed(self):
        """Committed count of the newest ring (a pure memory read; meters/wake probes)."""
        r = self.current
        return r.committed() if r is not None else 0

    def latest(self):
        """(reader, index) of the newest committed frame -- never lappable -- or None."""
        for r in reversed(self._readers):
            n = r.committed()
            if n > 0:
                return r, n - 1
        return None

    def drain(self, limit=None):
        """Yield (reader, index) for every not-yet-drained committed frame, in order, across
        ring changes. Lapped frames are skipped and counted in .lost. The consumer may itself
        hit Lapped inside a view for a frame we yielded -- it should count/skip and continue."""
        out = 0
        for k, r in enumerate(list(self._readers)):
            i = self._drained.get(r.ident, 0)
            first = r.first_available()
            if i < first:
                self.lost += first - i
                i = first
            n = r.committed()
            while i < n:
                yield r, i
                i += 1
                self._drained[r.ident] = i
                out += 1
                if limit is not None and out >= limit:
                    return
                first = r.first_available()
                if i < first:
                    self.lost += first - i
                    i = first
                n = r.committed()
            self._drained[r.ident] = i
            if not r.finalized() or k == len(self._readers) - 1:
                return                        # live ring drained dry (or last known ring)

    def drained_through(self, reader):
        """The drain cursor for a ring (how far a sequential consumer has consumed)."""
        return self._drained.get(reader.ident, 0)

    def skip_to_latest(self):
        """Start the drain cursor at NOW: everything already committed is history, not backlog
        -- skipped silently, not counted as lost. For consumers that begin at a moment in time
        (the recorder starts at the click; old frames are deliberately not archived)."""
        self.poll()
        for r in self._readers:
            self._drained[r.ident] = max(self._drained.get(r.ident, 0), r.committed())

    def ended(self):
        return self._ended

    def close(self):
        for r in self._readers:
            r.close()
        self._readers = []
        if self._head is not None:
            self._head.close()
            self._head = None


# --------------------------------- latest-wins slot ----------------------------------------


class LatestSlot:
    """A latest-wins record slot in shared memory (the backend's state channel: written ~20 Hz,
    where a per-frame LOG would be waste and a blocked file append stalls the CONTROL LOOP --
    the loop that has to keep processing estop). Layout: [seq u64 | len u32 | pad | payload].
    Single writer, seqlock: seq goes odd during a write, even after; readers retry on odd or
    changed. version() = seq//2 = how many writes ever (consumers meter on deltas). Payload is
    JSON -- one smallish record, parse cost irrelevant; what matters is that write() is pure
    memory stores."""

    _HDR = struct.Struct('<QII')

    def __init__(self, name=None, cap=1 << 16, create=False):
        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=self._HDR.size + cap,
                                                   name=name or uuid.uuid4().hex)
            self._shm.buf[:self._HDR.size] = self._HDR.pack(0, 0, 0)
        else:
            try:
                self._shm = shared_memory.SharedMemory(name=name, create=False)
            except (FileNotFoundError, OSError) as e:
                raise ValueError(f"state slot {name} gone") from e
        self.name = self._shm.name
        self._cap = self._shm.size - self._HDR.size

    def write(self, obj):
        buf = self._shm.buf
        payload = json.dumps(obj, separators=(',', ':')).encode('utf-8')
        if len(payload) > self._cap:
            raise ValueError(f"state record {len(payload)}B exceeds slot cap {self._cap}B")
        seq = struct.unpack_from('<Q', buf, 0)[0]
        struct.pack_into('<Q', buf, 0, seq + 1)              # odd: mid-write
        buf[self._HDR.size:self._HDR.size + len(payload)] = payload
        struct.pack_into('<I', buf, 8, len(payload))
        struct.pack_into('<Q', buf, 0, seq + 2)              # even: committed

    def read(self):
        """(version, record) of the latest committed write, or None if none/torn (retry next
        poll -- the writer is mid-store for microseconds at most)."""
        buf = self._shm.buf
        for _ in range(4):
            seq = struct.unpack_from('<Q', buf, 0)[0]
            if seq == 0 or seq & 1:
                if seq == 0:
                    return None
                continue
            n = struct.unpack_from('<I', buf, 8)[0]
            raw = bytes(buf[self._HDR.size:self._HDR.size + n])
            if struct.unpack_from('<Q', buf, 0)[0] == seq:
                try:
                    return seq // 2, json.loads(raw.decode('utf-8'))
                except ValueError:
                    return None                              # torn beyond repair this tick
        return None

    def close(self):
        self._shm.close()
