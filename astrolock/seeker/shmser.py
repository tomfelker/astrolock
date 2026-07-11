"""
Shared-memory .ser segments: the live-IPC frame path that never touches the disk.

Sustained full-rate .ser writes (~400-500 MB/s on the real rig) exhaust a consumer SSD's
erased-block pool and the whole pipeline hitches (see docs/notes.md, 2026-07-10). But the live
pipeline only ever reads recent frames -- the disk was just a bus. So an idle (non-recording)
segment's frames live in a named shared-memory section laid out exactly like the .ser file
would be: header, then frames appended linearly. NO ring buffer, no overwrites -- we simply
roll to a NEW section exactly where we'd have rolled to a new .ser file (shm segments just
roll sooner, since a segment is committed RAM). A written frame is immutable, so there is
nothing to tear and nothing to lock; the writer publishes a total-frames counter after each
frame and readers gate on it. Old sections evaporate when the writer and last reader close --
Windows named sections are kernel refcounted, session-local, and same-user by default DACL
(a ramdrive that auto-deletes on last close). Recording ('important') segments still go to
disk as plain .ser: a pass is a bounded burst, which SSDs handle; it's the *indefinite* idle
streaming they can't.

Discovery keeps every glob/rollover path working: the segment's .ser exists on disk as a
178-byte MARKER -- just the SER header, with 'SHMSER:<frame cap>' in the instrument field --
and SerReader transparently redirects to the section named by the file's basename. (Later:
glob the jsonl sidecars instead and let each name its backing file/section.)

Region layout:    [SER header][frame 0 .. cap-1 slots, appended in order][total u64]

Reads currently copy out of the section (simple, and close() stays safe); in-place zero-copy
views are a straightforward later optimization since committed frames never change.
"""

import os
from multiprocessing import shared_memory

import numpy as np

from astrolock.seeker import ser as ser_mod

MARKER_PREFIX = b'SHMSER:'


def is_shm_header(header):
    """True if a parsed SerHeader is a shm marker (instrument = 'SHMSER:<frame cap>')."""
    return bytes(header.instrument).startswith(MARKER_PREFIX)


def _frame_cap(header):
    return int(bytes(header.instrument)[len(MARKER_PREFIX):].split(b'\x00')[0])


def _section_name(path):
    """The section is named by the marker's basename -- unique per segment (timestamped), so an
    incompatible format change (size/depth) rolls to a new segment = a new section, like files."""
    return os.path.basename(str(path))


class _Views:
    """Numpy views over a mapped region: the per-frame slots and the published-total counter."""

    def __init__(self, buf, header, cap):
        bpf = ser_mod.bytes_per_frame(header)
        dtype = np.dtype('u1') if ser_mod.bytes_per_channel(header.pixel_depth_per_plane) == 1 \
            else np.dtype('<u2')
        nch = ser_mod.num_channels_for_color_id(header.color_id)
        shape = ((cap, header.image_height, header.image_width) if nch == 1
                 else (cap, header.image_height, header.image_width, nch))
        self.frames = np.ndarray(shape, dtype=dtype, buffer=buf, offset=ser_mod.HEADER_SIZE)
        self.total = np.ndarray((1,), dtype='<u8', buffer=buf,
                                offset=ser_mod.HEADER_SIZE + cap * bpf)


class ShmSerWriter:
    """Drop-in for SerWriter (write_frame/close/frame_count) writing frames into a shared-memory
    section; the only disk artifact is the 178-byte marker header at ``path``. Capacity is fixed
    at ``cap`` frames (it's committed RAM) -- the cam rolls segments at or before that, exactly
    as it rolls files."""

    def __init__(self, path, width, height, color_id=ser_mod.ColorId.MONO,
                 pixel_depth_per_plane=16, cap=128, observer='', telescope=''):
        self.path = str(path)
        self.cap = int(cap)
        self.frame_count = 0
        self.header = ser_mod.SerHeader(
            file_id=ser_mod.FILE_ID, lu_id=0, color_id=ser_mod.ColorId(color_id),
            little_endian=ser_mod.LITTLE_ENDIAN_FLAG, image_width=width, image_height=height,
            pixel_depth_per_plane=pixel_depth_per_plane, frame_count=ser_mod.SENTINEL_FRAME_COUNT,
            observer=observer, instrument=f"SHMSER:{self.cap}", telescope=telescope,
            date_time=0, date_time_utc=0)
        self._bytes_per_frame = ser_mod.bytes_per_frame(self.header)
        hdr = ser_mod.pack_header(self.header)
        with open(self.path, 'wb') as f:                 # the discovery marker (no handle kept open)
            f.write(hdr)
        size = ser_mod.HEADER_SIZE + self.cap * self._bytes_per_frame + 8
        self._shm = shared_memory.SharedMemory(name=_section_name(self.path), create=True, size=size)
        self._shm.buf[:ser_mod.HEADER_SIZE] = hdr        # region is header-true, like the file would be
        self._v = _Views(self._shm.buf, self.header, self.cap)
        self._v.total[0] = 0
        self._dtype = self._v.frames.dtype

    def write_frame(self, frame, t_utc=None):            # t_utc accepted for SerWriter parity (no trailer)
        if self.frame_count >= self.cap:
            raise ValueError(f"shm segment full ({self.cap} frames); roll to a new segment")
        arr = np.ascontiguousarray(frame, dtype=self._dtype)
        if arr.nbytes != self._bytes_per_frame:
            raise ValueError(f"frame is {arr.nbytes} bytes, expected {self._bytes_per_frame}")
        self._v.frames[self.frame_count][...] = arr.reshape(self._v.frames.shape[1:])
        self.frame_count += 1
        self._v.total[0] = self.frame_count               # published AFTER the bytes: the commit point

    def close(self):
        """Patch the marker's frame count (finalize semantics, like SerWriter) and drop our handle;
        the section lives on exactly as long as some reader still holds it."""
        try:
            with open(self.path, 'r+b') as f:
                f.seek(ser_mod.FRAME_COUNT_OFFSET)
                f.write(np.int32(self.frame_count).tobytes())
        except OSError:
            pass                                          # marker already cleaned up -> nothing to patch
        self._v = None                                    # release our buffer exports before close()
        self._shm.close()


class ShmSerReader:
    """Reader side, constructed by SerReader when it opens a marker file. Same contract:
    frames_total() and read_frame(index) -> read-only array, IndexError when not yet committed.
    Committed frames are immutable, so there are no torn reads and no locks."""

    def __init__(self, path, header):
        self.header = header
        self.cap = _frame_cap(header)
        try:
            self._shm = shared_memory.SharedMemory(name=_section_name(path), create=False)
        except (FileNotFoundError, OSError) as e:
            raise ValueError(f"{path}: shm segment is gone (writer and all readers exited)") from e
        self._v = _Views(self._shm.buf, header, self.cap)

    def frames_total(self):
        return int(self._v.total[0])

    def read_frame(self, index, to_float=False):
        total = self.frames_total()
        if index < 0 or index >= total:
            raise IndexError(f"frame {index} not available (have {total})")
        out = np.array(self._v.frames[index])             # copy out of the shared region
        out.setflags(write=False)                         # SerReader parity: read-only frames
        if to_float:
            out = out.astype(np.float32) / ser_mod.container_max(self.header.pixel_depth_per_plane)
        return out

    def close(self):
        self._v = None                                    # release buffer exports before close()
        self._shm.close()
