"""
Debayer correctness. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_bayer
"""

import numpy as np

from astrolock.seeker import bayer, ser


def _mosaic():
    # ul=10 (col0,row0), ur=20 (col1,row0), ll=30 (row1,col0), lr=40 (row1,col1)
    m = np.zeros((4, 4), np.uint16)
    m[0::2, 0::2] = 10
    m[0::2, 1::2] = 20
    m[1::2, 0::2] = 30
    m[1::2, 1::2] = 40
    return m


def test_debayer_rggb():
    rgb = bayer.debayer_to_rgb(_mosaic(), ser.ColorId.BAYER_RGGB)
    assert rgb.shape == (2, 2, 3)
    assert np.allclose(rgb[..., 0], 10)      # R = ul
    assert np.allclose(rgb[..., 1], 25)      # G = (ur+ll)/2 = (20+30)/2
    assert np.allclose(rgb[..., 2], 40)      # B = lr


def test_debayer_bggr():
    rgb = bayer.debayer_to_rgb(_mosaic(), ser.ColorId.BAYER_BGGR)
    assert np.allclose(rgb[..., 0], 40)      # R = lr
    assert np.allclose(rgb[..., 1], 25)      # G = (ur+ll)/2
    assert np.allclose(rgb[..., 2], 10)      # B = ul


def test_mono_sum_and_is_bayer():
    assert np.allclose(bayer.to_mono_sum(_mosaic()), 100)   # 10+20+30+40
    assert bayer.is_bayer(ser.ColorId.BAYER_GRBG)
    assert not bayer.is_bayer(ser.ColorId.MONO)


if __name__ == '__main__':
    test_debayer_rggb()
    test_debayer_bggr()
    test_mono_sum_and_is_bayer()
    print("test_bayer: OK")
