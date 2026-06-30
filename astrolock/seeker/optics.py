"""
Sensor + optics database and field-of-view helpers for AstroLock Seeker.

The facts in ``data/optics_db.json`` are vendored from Stellarium's Oculars plugin
(``default_ocular.ini``) -- bare facts (chip resolution / pixel pitch, focal length / aperture);
Stellarium itself is GPL. Source:
https://github.com/Stellarium/stellarium/blob/master/plugins/Oculars/resources/default_ocular.ini

Use this to pick a sensor + optic and get the field of view and plate scale, e.g. when bringing a
new (main) camera online. The math is plain scalar trig: for ``N`` pixels at pitch ``p`` (um) on an
optic of focal length ``f`` (mm),

    chip_mm = N * p / 1000;  fov = 2*atan(chip_mm / (2*f));  scale = 206.265 * p / f  (arcsec/px).

A focal reducer / barlow scales the effective focal length by its multiplier (<1 reducer = wider
field, >1 barlow = narrower). ``arcsec_per_px`` converts straight to the tracker's ``rad_per_px``.

    python -m astrolock.seeker.optics                                  # list everything
    python -m astrolock.seeker.optics --sensor "EOS 5D (Full Frame)" --optic "Celestron C8 f/10"
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'optics_db.json')

_ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0     # 206264.8


@dataclass
class Sensor:
    name: str
    res_x: int
    res_y: int
    pixel_um: float
    bayer: str = None        # Bayer mosaic pattern (e.g. 'RGGB') for a color sensor; None = mono

    @property
    def is_color(self):
        return self.bayer is not None

    @property
    def chip_w_mm(self):
        return self.res_x * self.pixel_um / 1000.0

    @property
    def chip_h_mm(self):
        return self.res_y * self.pixel_um / 1000.0


@dataclass
class Optic:
    name: str
    focal_length_mm: float
    aperture_mm: float


def load_db(path=None):
    """-> (sensors, optics, reducers): name-keyed dicts of Sensor, Optic, and float multipliers."""
    with open(path or _DEFAULT_DB) as f:
        raw = json.load(f)
    sensors = {s['name']: Sensor(s['name'], s['res_x'], s['res_y'], s['pixel_um'], s.get('bayer'))
               for s in raw['sensors']}
    optics = {o['name']: Optic(o['name'], o['focal_length_mm'], o['aperture_mm'])
              for o in raw['optics']}
    reducers = {r['name']: float(r['multiplier']) for r in raw['reducers']}
    return sensors, optics, reducers


def arcsec_per_px(pixel_um, focal_length_mm):
    """Plate scale: arcsec per pixel for pixel pitch ``pixel_um`` at focal length ``focal_length_mm``."""
    return _ARCSEC_PER_RAD * (pixel_um / 1000.0) / focal_length_mm


def rad_per_px(pixel_um, focal_length_mm):
    """Same as arcsec_per_px but in radians/px -- the tracker's pixel-scale tunable."""
    return (pixel_um / 1000.0) / focal_length_mm        # small-angle: chip_mm_per_px / f


def fov_deg(sensor, focal_length_mm):
    """(fov_x, fov_y) degrees for a Sensor on an optic of the given (effective) focal length."""
    fx = math.degrees(2.0 * math.atan(sensor.chip_w_mm / (2.0 * focal_length_mm)))
    fy = math.degrees(2.0 * math.atan(sensor.chip_h_mm / (2.0 * focal_length_mm)))
    return fx, fy


def configuration(sensor, optic, reducer=1.0):
    """A dict summary of a sensor+optic(+reducer) combo: effective focal length, FoV, plate scale."""
    f = optic.focal_length_mm * reducer
    fx, fy = fov_deg(sensor, f)
    return {
        'sensor': sensor.name,
        'optic': optic.name,
        'reducer': reducer,
        'effective_focal_mm': f,
        'fov_x_deg': fx,
        'fov_y_deg': fy,
        'arcsec_per_px': arcsec_per_px(sensor.pixel_um, f),
        'rad_per_px': rad_per_px(sensor.pixel_um, f),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Sensor/optics DB + field-of-view helper")
    p.add_argument('--db', default=None, help="path to optics_db.json (default: vendored)")
    p.add_argument('--sensor', default=None, help="sensor name (exact)")
    p.add_argument('--optic', default=None, help="optic name (exact)")
    p.add_argument('--reducer', default=None, help="reducer/barlow name (exact), optional")
    args = p.parse_args(argv)
    sensors, optics, reducers = load_db(args.db)

    if not (args.sensor and args.optic):
        print("sensors:")
        for s in sensors.values():
            print(f"  {s.name:28s} {s.res_x}x{s.res_y}  {s.pixel_um} um")
        print("optics:")
        for o in optics.values():
            print(f"  {o.name:28s} f={o.focal_length_mm}mm  aperture={o.aperture_mm}mm")
        print("reducers:")
        for n, m in reducers.items():
            print(f"  {n:28s} x{m}")
        print("\npass --sensor and --optic (and optionally --reducer) for a field-of-view summary.")
        return

    if args.sensor not in sensors:
        raise SystemExit(f"unknown sensor {args.sensor!r}")
    if args.optic not in optics:
        raise SystemExit(f"unknown optic {args.optic!r}")
    mult = reducers[args.reducer] if args.reducer else 1.0
    if args.reducer and args.reducer not in reducers:
        raise SystemExit(f"unknown reducer {args.reducer!r}")
    c = configuration(sensors[args.sensor], optics[args.optic], mult)
    print(f"{c['sensor']}  +  {c['optic']}" + (f"  (x{mult})" if mult != 1.0 else ""))
    print(f"  effective focal length : {c['effective_focal_mm']:.1f} mm")
    print(f"  field of view          : {c['fov_x_deg']:.3f} x {c['fov_y_deg']:.3f} deg")
    print(f"  plate scale            : {c['arcsec_per_px']:.3f} arcsec/px")


if __name__ == '__main__':
    main()
