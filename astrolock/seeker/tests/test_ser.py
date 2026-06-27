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


if __name__ == '__main__':
    test_ser_roundtrip()
    test_ser_12bit_depth()
    print("test_ser: OK")
