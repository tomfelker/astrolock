"""
Optics DB load + field-of-view helpers. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_optics
"""

from astrolock.seeker import optics


def test_db_loads():
    sensors, opts, reducers = optics.load_db()
    assert len(sensors) >= 40 and len(opts) >= 6 and len(reducers) == 2
    # a mono ZWO chip: is_color reads False, no Bayer pattern
    s = sensors["ZWO ASI678MM"]
    assert (s.res_x, s.res_y, s.pixel_um) == (3840, 2160, 2.0)
    assert s.is_color is False and s.bayer is None
    assert abs(s.chip_w_mm - 3840 * 2.0 / 1000.0) < 1e-9
    assert opts["Celestron CPC 1100"].focal_length_mm == 2800
    assert reducers["Celestron f/6.3 Reducer/Corrector"] == 0.63
    # the Seeker rig defaults (backend.py) must resolve
    s = sensors["ZWO ASI678MC"]
    assert s.res_x == 3840 and s.pixel_um == 2.0 and s.bayer == "RGGB" and s.is_color is True
    assert opts["8mm CS f/1.4"].focal_length_mm == 8 and opts["Celestron CPC 1100"].focal_length_mm == 2800


def test_plate_scale_and_fov():
    # The standard rule: arcsec/px = 206.265 * pixel_um / focal_mm.
    assert abs(optics.arcsec_per_px(5.0, 1000.0) - 1.0313) < 1e-3
    assert abs(optics.arcsec_per_px(5.0, 1000.0) - optics.rad_per_px(5.0, 1000.0) * optics._ARCSEC_PER_RAD) < 1e-9

    # A 36 mm-wide chip (7200 px * 5 um) at f=1000 mm -> 2*atan(36/2000) ~ 2.062 deg.
    sensor = optics.Sensor("test", 7200, 4800, 5.0)
    fx, fy = optics.fov_deg(sensor, 1000.0)
    assert abs(fx - 2.0624) < 0.01
    assert fx > fy > 0                       # wider than tall


def test_reducer_widens_barlow_narrows():
    sensors, opts, reducers = optics.load_db()
    s, o = sensors["ZWO ASI678MC"], opts["Celestron CPC 1100"]
    base = optics.configuration(s, o)['fov_x_deg']
    wide = optics.configuration(s, o, reducers["Celestron f/6.3 Reducer/Corrector"])['fov_x_deg']   # 0.63 -> wider
    narrow = optics.configuration(s, o, reducers["Celestron X-Cel LX 3x Barlow"])['fov_x_deg']      # 3.0 -> narrower
    assert wide > base > narrow


if __name__ == '__main__':
    test_db_loads()
    test_plate_scale_and_fov()
    test_reducer_widens_barlow_narrows()
    print("test_optics: OK")
