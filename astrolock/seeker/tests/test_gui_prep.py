"""
GUI image-prep + overlay drawing, headless (no dearpygui needed -- it's imported lazily
inside gui.main). Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_gui_prep
"""

import numpy as np

from astrolock.seeker import gui, ser


def test_prepare_mono_and_box():
    fr = np.zeros((8, 8), np.uint16)
    fr[4, 4] = 60000
    w, h, rgba = gui.prepare_rgba(fr, ser.ColorId.MONO, 2.2)
    assert (w, h) == (8, 8)
    assert rgba.shape == (8, 8, 4)
    gui.draw_box(rgba, 4, 4, 2, gui._MOVING)          # box edges at rows/cols 2 and 6
    assert np.allclose(rgba[2, 4][:3], gui._MOVING[:3])
    assert np.allclose(rgba[6, 4][:3], gui._MOVING[:3])


def test_prepare_bayer_is_half_res():
    m = np.zeros((8, 8), np.uint16)
    m[0::2, 0::2] = 1000          # red sites
    w, h, rgba = gui.prepare_rgba(m, ser.ColorId.BAYER_RGGB, 2.2, (1.24, 1.98))
    assert (w, h) == (4, 4)        # debayered to half resolution
    assert rgba.shape == (4, 4, 4)


if __name__ == '__main__':
    test_prepare_mono_and_box()
    test_prepare_bayer_is_half_res()
    print("test_gui_prep: OK")
