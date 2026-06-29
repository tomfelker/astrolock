"""
Optics DB load + field-of-view helpers. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_optics
"""

from astrolock.seeker import optics


def test_db_loads():
    sensors, opts, reducers = optics.load_db()
    assert len(sensors) == 17 and len(opts) == 22 and len(reducers) == 3
    s = sensors["EOS 5D (Full Frame)"]
    assert (s.res_x, s.res_y, s.pixel_um) == (4368, 2912, 8.2)
    assert abs(s.chip_w_mm - 4368 * 8.2 / 1000.0) < 1e-9
    assert opts["Celestron C8 f/10"].focal_length_mm == 2032
    assert reducers["F/6.3 Reducer"] == 0.63


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
    s, o = sensors["EOS 5D (Full Frame)"], opts["Celestron C8 f/10"]
    base = optics.configuration(s, o)['fov_x_deg']
    wide = optics.configuration(s, o, reducers["F/6.3 Reducer"])['fov_x_deg']    # 0.63 -> wider
    narrow = optics.configuration(s, o, reducers["Barlow 2x"])['fov_x_deg']      # 2.0 -> narrower
    assert wide > base > narrow


if __name__ == '__main__':
    test_db_loads()
    test_plate_scale_and_fov()
    test_reducer_widens_barlow_narrows()
    print("test_optics: OK")
