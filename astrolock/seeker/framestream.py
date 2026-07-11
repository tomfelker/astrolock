"""
Named rolling frame streams: the one interface every producer/consumer of video uses.

A stream is a sequence of segments; each segment is a .frames.jsonl sidecar (the commit spine:
line N commits frame N) plus a pixel store -- a plain .ser file on disk, or a same-named
shared-memory section (see shmser.py) that never touches the disk. Discovery is BY SIDECAR:
consumers glob '*_<name>.frames.jsonl' and derive the segment identity ('<stem>.ser', which for
a shm segment is just a name -- no file exists). Each sidecar record carries 'store': 'ser'|'shm',
and open_reader() resolves the store FROM THE SIDECAR -- the records are the authority, not file
existence. (Future stores -- e.g. mjpeg with variable-size frames -- would add per-record
offset/size fields; the spine design already permits it.)

Write side: FrameStream owns the writer + sidecar pair and the roll bookkeeping. Committing is
two-phase to preserve the cam's contract (pixels flushed, optional wait, THEN the commit line):

    stream.open_segment(seg_ts, w, h, color_id, depth, shm=..., cap=...)
    stream.write_pixels(frame); ...; stream.commit(index=i, ...)     # or stream.write(frame, ...)

Rolling a shm segment retains the PREVIOUS segment's section handle until the roll after next:
a section dies with its last handle, so without retention a reader that hadn't attached yet
would race the roll and lose the tail of the old segment.
"""

import glob
import json
import os

from astrolock.seeker import ser as ser_mod
from astrolock.seeker import shmser
from astrolock.seeker.sidecar import JsonlWriter


def sidecar_glob(session_dir, name):
    """Sorted sidecar paths for stream `name` ('guide', 'main_debug', ...) -- the discovery path
    (never glob .ser: a shm segment has no file)."""
    return sorted(glob.glob(os.path.join(str(session_dir), f'*_{name}.frames.jsonl')))


def ser_path_of(frames_path):
    """The segment identity: '<stem>.ser'. A real file for disk segments; for shm segments just
    a name (consumers use it as an ID and to derive sibling sidecar names)."""
    return str(frames_path)[:-len('.frames.jsonl')] + '.ser'


def frames_path_of(ser_path):
    return str(ser_path)[:-len('.ser')] + '.frames.jsonl'


def segment_paths(session_dir, name):
    """Sorted segment identities (.ser paths, possibly virtual) for stream `name`."""
    return [ser_path_of(p) for p in sidecar_glob(session_dir, name)]


def open_reader(ser_path):
    """Open a segment for reading by its identity. The SIDECAR specifies the store: its records
    carry 'store': 'ser'|'shm', so read the first committed record and open the .ser file or
    attach the same-named shared-memory section accordingly (records without the field are old
    plain-file recordings). No committed record yet -> the store is unknown: fall back to a
    growing .ser file if one exists (pre-first-commit disk segment), else raise ValueError --
    callers already treat ValueError as 'not ready / gone'."""
    first = ''
    try:
        with open(frames_path_of(ser_path), 'r', encoding='utf-8') as f:
            first = f.readline()
    except (FileNotFoundError, OSError):
        pass
    if first.endswith('\n'):
        store = json.loads(first).get('store', 'ser')
        if store == 'shm':
            return shmser.ShmSerReader(ser_path)
        return ser_mod.SerReader(ser_path)
    if os.path.exists(ser_path):
        return ser_mod.SerReader(ser_path)                 # disk segment, nothing committed yet
    raise ValueError(f"{ser_path}: no committed frames yet (store unknown)")


class FrameStream:
    """Write side of a named stream: one open segment at a time, rolled by the caller."""

    def __init__(self, out_dir, name):
        self.out_dir, self.name = str(out_dir), str(name)
        self._writer = None
        self._sidecar = None
        self._retired = None                  # previous segment's writer, kept to bridge reader attach
        self._store = 'ser'
        self.ser_path = None
        self.frames_path = None

    @property
    def frame_count(self):
        return self._writer.frame_count if self._writer is not None else 0

    def open_segment(self, seg_ts, width, height, color_id=ser_mod.ColorId.MONO,
                     pixel_depth_per_plane=16, shm=False, cap=64):
        """Finalize any current segment and open a fresh one named '<seg_ts>_<name>'."""
        self._roll_out()
        stem = os.path.join(self.out_dir, f"{seg_ts}_{self.name}")
        self.ser_path, self.frames_path = stem + '.ser', stem + '.frames.jsonl'
        self._store = 'shm' if shm else 'ser'
        if shm:
            self._writer = shmser.ShmSerWriter(self.ser_path, width, height, color_id=color_id,
                                               pixel_depth_per_plane=pixel_depth_per_plane, cap=cap)
        else:
            self._writer = ser_mod.SerWriter(self.ser_path, width, height, color_id=color_id,
                                             pixel_depth_per_plane=pixel_depth_per_plane)
        self._sidecar = JsonlWriter(self.frames_path)

    def write_pixels(self, frame, t_utc=None):
        """Phase 1: flush the pixels (commit them with commit(); frame N = the Nth commit)."""
        self._writer.write_frame(frame, t_utc=t_utc)

    def commit(self, **record):
        """Phase 2: append the commit line (with the self-describing 'store' field)."""
        self._sidecar.append({**record, 'store': self._store})

    def write(self, frame, t_utc=None, **record):
        """One-shot write+commit for producers with no between-phase work (debug/focus streams)."""
        self.write_pixels(frame, t_utc=t_utc)
        self.commit(**record)

    def _roll_out(self):
        """Finalize the open segment; retain a shm segment's handle for one more roll."""
        if self._sidecar is not None:
            self._sidecar.close()
            self._sidecar = None
        if self._retired is not None:
            self._retired.close()
            self._retired = None
        if self._writer is not None:
            if isinstance(self._writer, shmser.ShmSerWriter):
                self._writer.finalize()
                self._retired = self._writer   # keep the section alive across the reader-attach race
            else:
                self._writer.close()
            self._writer = None

    def close(self):
        self._roll_out()
        if self._retired is not None:
            self._retired.close()
            self._retired = None
