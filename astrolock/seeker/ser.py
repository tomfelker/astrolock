"""
Self-contained SER reader + writer for AstroLock Seeker.

The SER format is trivial: a 178-byte little-endian header followed by raw frame pixels
back to back. We carry our own reader/writer (no tensorez dependency); the format logic
is adapted from the public SER spec.

Seeker-specific conventions (see astrolock_seeker.md):

- While recording, ``frame_count`` is written as a sentinel (INT_MAX) and patched to the
  true count on clean close. **Readers must never trust the header count** -- compute the
  number of complete frames from the file size instead (``frames_on_disk``). This keeps a
  growing file readable and survives a crash mid-capture.
- Byte order: we write native little-endian pixels and set the ``little_endian`` header
  field so our own reader interprets it correctly. (The SER spec's flag is famously
  inverted; we just stay self-consistent. Interop with third-party SER players is a later
  concern.)
"""

import collections
import enum
import struct

import numpy as np

# '<' little-endian, standard sizes, no alignment padding.
#   file_id(14s) lu_id(l) color_id(l) little_endian(l) image_width(l) image_height(l)
#   pixel_depth_per_plane(l) frame_count(l) observer(40s) instrument(40s) telescope(40s)
#   date_time(q) date_time_utc(q)
HEADER_STRUCT = struct.Struct('<14slllllll40s40s40sqq')
HEADER_SIZE = HEADER_STRUCT.size
assert HEADER_SIZE == 178, f"unexpected SER header size {HEADER_SIZE}"

# Byte offset of the frame_count field within the header: 14s + 6 longs.
FRAME_COUNT_OFFSET = 14 + 4 * 6

# Sentinel written while a capture is in progress (signed 32-bit max).
SENTINEL_FRAME_COUNT = 0x7FFFFFFF

FILE_ID = b'LUCAM-RECORDER'  # 14 bytes, the standard SER file id

SerHeader = collections.namedtuple(
    'SerHeader',
    'file_id lu_id color_id little_endian image_width image_height '
    'pixel_depth_per_plane frame_count observer instrument telescope '
    'date_time date_time_utc'
)


class ColorId(enum.IntEnum):
    MONO = 0
    BAYER_RGGB = 8
    BAYER_GRBG = 9
    BAYER_GBRG = 10
    BAYER_BGGR = 11
    RGB = 100
    BGR = 101


# Our own little-endian flag value (see module docstring re: the spec's inversion).
LITTLE_ENDIAN_FLAG = 0


def num_channels_for_color_id(color_id):
    return 3 if int(color_id) >= ColorId.RGB else 1


def _fixed(s, n):
    """Encode a str/bytes to exactly n bytes (truncated/zero-padded)."""
    if isinstance(s, str):
        s = s.encode('ascii', errors='replace')
    return s[:n].ljust(n, b'\x00')


def pack_header(header):
    return HEADER_STRUCT.pack(
        _fixed(header.file_id, 14),
        header.lu_id, int(header.color_id), header.little_endian,
        header.image_width, header.image_height,
        header.pixel_depth_per_plane, header.frame_count,
        _fixed(header.observer, 40), _fixed(header.instrument, 40), _fixed(header.telescope, 40),
        header.date_time, header.date_time_utc,
    )


def unpack_header(raw):
    return SerHeader._make(HEADER_STRUCT.unpack(raw))


def bytes_per_channel(pixel_depth_per_plane):
    """SER stores 1 byte/channel when depth <= 8, otherwise 2 (the value spans up to 16 bits)."""
    return 1 if pixel_depth_per_plane <= 8 else 2


def container_max(pixel_depth_per_plane):
    """
    Full-scale value for normalization. We normalize by the *container* (255 or 65535), not
    2**depth: cameras like the ASI in RAW16 scale the ADC to fill the 16-bit range (proven
    empirically -- peaks reach ~65535, not a 12-bit 4095), so depth is informational
    precision, not the value scale.
    """
    return (1 << (8 * bytes_per_channel(pixel_depth_per_plane))) - 1


def _numpy_dtype(pixel_depth_per_plane, little_endian_flag):
    if bytes_per_channel(pixel_depth_per_plane) == 1:
        return np.dtype('uint8')
    # Match the reader convention used historically: flag==0 -> little-endian.
    dt = np.dtype('uint16')
    return dt.newbyteorder('<' if little_endian_flag == 0 else '>')


def bytes_per_frame(header):
    n_ch = num_channels_for_color_id(header.color_id)
    return bytes_per_channel(header.pixel_depth_per_plane) * n_ch * header.image_width * header.image_height


class SerWriter:
    """
    Append frames to a .ser file. Writes a sentinel frame_count while open and patches it
    on close(). flush() after every frame so other processes can read the bytes via the
    page cache immediately (the commit point; see astrolock_seeker.md).

    Usable as a context manager.
    """

    def __init__(self, path, width, height, color_id=ColorId.MONO,
                 pixel_depth_per_plane=16, observer='', instrument='', telescope=''):
        self.path = str(path)
        self.width = width
        self.height = height
        self.color_id = ColorId(color_id)
        self.pixel_depth_per_plane = pixel_depth_per_plane
        self.frame_count = 0
        self._closed = False

        self.header = SerHeader(
            file_id=FILE_ID, lu_id=0, color_id=self.color_id,
            little_endian=LITTLE_ENDIAN_FLAG, image_width=width, image_height=height,
            pixel_depth_per_plane=pixel_depth_per_plane, frame_count=SENTINEL_FRAME_COUNT,
            observer=observer, instrument=instrument, telescope=telescope,
            date_time=0, date_time_utc=0,
        )
        self._bytes_per_frame = bytes_per_frame(self.header)
        self._dtype = _numpy_dtype(pixel_depth_per_plane, LITTLE_ENDIAN_FLAG)

        self._file = open(self.path, 'wb')
        self._file.write(pack_header(self.header))
        self._file.flush()

    def write_frame(self, frame):
        """
        frame: ndarray shaped (height, width) for mono or (height, width, channels).
        Converted to the writer's dtype if needed. Bytes are flushed before returning.
        """
        arr = np.ascontiguousarray(frame, dtype=self._dtype)
        if arr.nbytes != self._bytes_per_frame:
            raise ValueError(
                f"frame is {arr.nbytes} bytes, expected {self._bytes_per_frame} "
                f"({self.width}x{self.height}, depth {self.pixel_depth_per_plane})")
        self._file.write(arr.tobytes())
        self._file.flush()
        self.frame_count += 1
        return self.frame_count - 1  # index of the frame just written

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            # Patch the real frame count into the header.
            self._file.seek(FRAME_COUNT_OFFSET)
            self._file.write(struct.pack('<l', self.frame_count))
            self._file.flush()
        finally:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SerReader:
    """
    Read frames from a .ser file (live/growing or finalized). Always derives the available
    frame count from the file size via frames_on_disk(); the header count is ignored.

    By default each ``read_frame`` returns a **zero-copy** view onto the OS page cache via a
    per-frame ``np.memmap`` -- for our large frames this skips the page-cache->buffer copy that
    seek+read does and hands the bytes straight to torch. Mapping per frame keeps it trivial for a
    growing file (each frame is an already-complete byte range, never past EOF) and ties the
    mapping's lifetime to the returned array (it unmaps when the array is GC'd). The returned array
    is **read-only** -- which matches the old ``np.frombuffer`` behaviour, so callers already copy
    before mutating. ``use_mmap=False`` (or an automatic fallback on any mmap error) reverts to
    positioned seek+read for portability.
    """

    def __init__(self, path, use_mmap=True):
        self.path = str(path)
        self.use_mmap = use_mmap
        self._file = open(self.path, 'rb')
        raw = self._file.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise ValueError(f"{self.path}: file too short to contain a SER header")
        self.header = unpack_header(raw)
        self.bytes_per_frame = bytes_per_frame(self.header)
        self.num_channels = num_channels_for_color_id(self.header.color_id)
        self._dtype = _numpy_dtype(self.header.pixel_depth_per_plane, self.header.little_endian)

    def frames_on_disk(self):
        """Number of *complete* frames currently present, from the file size."""
        import os
        size = os.fstat(self._file.fileno()).st_size
        if self.bytes_per_frame <= 0:
            return 0
        return max(0, (size - HEADER_SIZE) // self.bytes_per_frame)

    def _frame_shape(self):
        if self.num_channels == 1:
            return (self.header.image_height, self.header.image_width)
        return (self.header.image_height, self.header.image_width, self.num_channels)

    def read_frame(self, index, to_float=False):
        """
        Return the frame at ``index`` shaped (height, width[, channels]), read-only.
        Raises IndexError if that frame isn't fully on disk yet.
        """
        available = self.frames_on_disk()
        if index < 0 or index >= available:
            raise IndexError(f"frame {index} not available (have {available})")
        offset = HEADER_SIZE + index * self.bytes_per_frame

        arr = None
        if self.use_mmap:
            try:
                # np.memmap handles the page-aligned-offset rule internally and keeps the mapping
                # alive as the array's base; it unmaps when the returned array is dropped.
                arr = np.memmap(self.path, dtype=self._dtype, mode='r',
                                offset=offset, shape=self._frame_shape())
            except (OSError, ValueError):
                self.use_mmap = False            # portability fallback for this reader's lifetime
                arr = None

        if arr is None:
            self._file.seek(offset)
            buf = self._file.read(self.bytes_per_frame)
            if len(buf) < self.bytes_per_frame:
                raise IndexError(f"frame {index} truncated on disk")
            arr = np.frombuffer(buf, dtype=self._dtype).reshape(self._frame_shape())

        if to_float:
            arr = arr.astype(np.float32) / container_max(self.header.pixel_depth_per_plane)
        return arr

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
