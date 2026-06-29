"""
SER reader/writer round-trip. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_ser
"""

import os

import numpy as np

from astrolock.seeker import ser
from astrolock.seeker.tests._util import fresh_dir


def test_ser_roundtrip():
    out = fresh_dir('ser')
    path = os.path.join(out, 'roundtrip.ser')
    W, H, N = 64, 48, 5
    rng = np.random.default_rng(0)
    frames = [(rng.random((H, W)) * 65535).astype(np.uint16) for _ in range(N)]

    w = ser.SerWriter(path, W, H, color_id=ser.ColorId.MONO, pixel_depth_per_plane=16)
    for i in range(2):
        w.write_frame(frames[i])

    # A separate reader sees the growing file without reopening.
    r = ser.SerReader(path)
    assert r.frames_on_disk() == 2
    np.testing.assert_array_equal(r.read_frame(0), frames[0])
    np.testing.assert_array_equal(r.read_frame(1), frames[1])
    try:
        r.read_frame(2)
        assert False, "frame 2 should not be available yet"
    except IndexError:
        pass

    for i in range(2, N):
        w.write_frame(frames[i])
    assert r.frames_on_disk() == N
    np.testing.assert_array_equal(r.read_frame(N - 1), frames[N - 1])

    # Header carries the sentinel while open, the true count after close.
    assert ser.SerReader(path).header.frame_count == ser.SENTINEL_FRAME_COUNT
    w.close()
    assert ser.SerReader(path).header.frame_count == N

    f = r.read_frame(0, to_float=True)
    assert f.dtype == np.float32 and f.shape == (H, W) and 0.0 <= f.max() <= 1.0
    r.close()


def test_ser_12bit_depth():
    # A 12-bit-depth file must still store 2 bytes/pixel and normalize by the 16-bit
    # container (not 4095), since the data is stored full-range.
    out = fresh_dir('ser12')
    path = os.path.join(out, 'd12.ser')
    W, H = 8, 6
    frame = np.full((H, W), 60000, dtype=np.uint16)  # near full 16-bit
    with ser.SerWriter(path, W, H, color_id=ser.ColorId.MONO, pixel_depth_per_plane=12) as w:
        w.write_frame(frame)
    r = ser.SerReader(path)
    assert r.header.pixel_depth_per_plane == 12
    assert r.bytes_per_frame == W * H * 2          # 2 bytes/pixel despite depth 12
    np.testing.assert_array_equal(r.read_frame(0), frame)
    f = r.read_frame(0, to_float=True)
    assert abs(f.max() - 60000 / 65535) < 1e-4     # normalized by container, not 4095
    r.close()


def test_ser_mmap_matches_io():
    # On a finalized file the mmap path is exercised for real on every platform; its result must
    # be byte-identical to the seek+read fallback, and read-only.
    out = fresh_dir('ser_mmap')
    path = os.path.join(out, 'mm.ser')
    W, H, N = 32, 20, 4
    rng = np.random.default_rng(1)
    frames = [(rng.random((H, W)) * 65535).astype(np.uint16) for _ in range(N)]
    with ser.SerWriter(path, W, H, color_id=ser.ColorId.MONO, pixel_depth_per_plane=16) as w:
        for f in frames:
            w.write_frame(f)

    rm = ser.SerReader(path, use_mmap=True)
    rio = ser.SerReader(path, use_mmap=False)
    for i in range(N):
        a = rm.read_frame(i)
        b = rio.read_frame(i)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, frames[i])
        assert not a.flags.writeable                       # zero-copy view is read-only
        np.testing.assert_allclose(rm.read_frame(i, to_float=True),
                                   rio.read_frame(i, to_float=True))
    assert isinstance(rm.read_frame(0), np.memmap)         # the mmap path was actually taken
    rm.close()
    rio.close()


def test_ser_mmap_color_and_growing():
    # Multi-channel shape, and the (mmap-default) reader following a growing file -- which on some
    # platforms falls back to seek+read while the writer holds the file open; either way correct.
    out = fresh_dir('ser_mmap_color')
    path = os.path.join(out, 'rgb.ser')
    W, H, N = 16, 12, 3
    rng = np.random.default_rng(2)
    frames = [(rng.random((H, W, 3)) * 65535).astype(np.uint16) for _ in range(N)]
    w = ser.SerWriter(path, W, H, color_id=ser.ColorId.RGB, pixel_depth_per_plane=16)
    w.write_frame(frames[0])
    r = ser.SerReader(path)                                # mmap by default
    assert r.frames_on_disk() == 1
    a = r.read_frame(0)
    assert a.shape == (H, W, 3)
    np.testing.assert_array_equal(a, frames[0])
    for f in frames[1:]:
        w.write_frame(f)
    assert r.frames_on_disk() == N                         # follows the growing file
    np.testing.assert_array_equal(r.read_frame(N - 1), frames[N - 1])
    w.close()
    r.close()


if __name__ == '__main__':
    test_ser_roundtrip()
    test_ser_12bit_depth()
    test_ser_mmap_matches_io()
    test_ser_mmap_color_and_growing()
    print("test_ser: OK")
