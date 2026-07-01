"""
Single sky-simulation process: the one owner of "what is where in the sky, when".

It holds the mapping from *system time* (perf_counter_ns, shared across processes) to the *simulated*
UTC of the test pass -- chosen once, here, so nobody downstream ever sees simulated time or picks
their own epoch. It propagates every point source (Hipparcos stars via Skyfield, the target satellite
via SGP4, resolved into its body-point cloud in an LVLH attitude) and publishes their topocentric ENU
directions as piecewise-linear anchors, in system time, to a JSONL ephemeris (see ephemeris.py).

This replaces per-camera propagation. Previously each camera process did its own SGP4 + catalog pass
and timed the sky against its own start, so two cameras placed a fast satellite at slightly different
world-times and drifted (a constant, slew-rate-scaling, per-run-signed pointing offset). One process,
one clock fixes that -- and kills the once-a-second star recompute jump, since consumers now lerp.

Every source is a uniform point-target: a star is one point; the satellite is its 68-ish body points,
each an independent target ("ISS (ZARYA) point k/N"), so the extended shape falls out of the same
machinery, no special case. Density is the only difference (stars: tens of seconds between anchors;
satellite points: a fraction of a second).
"""

import argparse
import datetime
import math
import os
import time

import numpy as np

from astrolock.seeker import bodies
from astrolock.seeker.ephemeris import anchor_record
from astrolock.seeker.sidecar import JsonlWriter


def _enu_from_altaz(az_rad, alt_rad):
    """(az from north toward east, alt) -> unit East-North-Up direction(s). Matches skysim._enu."""
    ca = np.cos(alt_rad)
    return np.stack([ca * np.sin(az_rad), ca * np.cos(az_rad), np.sin(alt_rad)], axis=-1)


def _load_tle(path):
    """Return (name, line1, line2) from a TLE file (2- or 3-line)."""
    with open(path) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if len(lines) >= 3 and not lines[0].startswith('1 '):
        return lines[0].strip(), lines[1], lines[2]
    return 'SAT', lines[0], lines[1]


class SkyPublisher:
    def __init__(self, args):
        from skyfield.api import Loader, wgs84, EarthSatellite
        from skyfield.data import hipparcos
        from skyfield.starlib import Star

        loader = Loader(args.cache_dir)
        self.ts = loader.timescale()
        eph = loader('de421.bsp')
        self.topos = wgs84.latlon(args.lat, args.lon, elevation_m=args.elev)
        self.observer = eph['earth'] + self.topos
        self.earth_r = 6371000.0 + args.elev

        with loader.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
        df = df[(df['magnitude'] <= args.mag_limit)
                & df['ra_degrees'].notnull() & df['dec_degrees'].notnull()]
        self.stars = Star.from_dataframe(df)
        self.star_ids = [f"star:{int(h)}" for h in df.index.to_numpy()]
        self.star_mag = df['magnitude'].to_numpy().astype(float)

        self.sat = None
        if args.tle_file:
            name, l1, l2 = _load_tle(args.tle_file)
            self.sat = EarthSatellite(l1, l2, name, self.ts)
            self.body_pts = bodies.points_for_name(name).astype(float)     # (P, 3) body-frame metres
            npts = len(self.body_pts)
            self.sat_ids = [f"{name} point {p}/{npts}" for p in range(npts)]
            self.sat_mag = args.target_mag + 2.5 * math.log10(npts)        # split flux over the points

        self.epoch = datetime.datetime.fromisoformat(args.epoch.replace('Z', '+00:00'))
        self.perf0_ns = time.perf_counter_ns()                             # system time <-> sim epoch anchor

    def _sf_times(self, t_ns):
        """Skyfield Time for system-time anchors t_ns (array): sim UTC = epoch + (t - perf0)."""
        secs = (np.asarray(t_ns, dtype=np.float64) - self.perf0_ns) * 1e-9
        e = self.epoch
        return self.ts.utc(e.year, e.month, e.day, e.hour, e.minute,
                           e.second + e.microsecond * 1e-6 + secs)

    def star_dirs(self, t_ns):
        """(len(t_ns), N_stars, 3) ENU dirs of every star at each anchor time."""
        out = []
        for t in t_ns:                                    # a few anchors per emit; catalog pass each
            alt, az, _ = self.observer.at(self._sf_times([t])).observe(self.stars).apparent().altaz()
            out.append(_enu_from_altaz(np.asarray(az.radians).reshape(-1),
                                       np.asarray(alt.radians).reshape(-1)))
        return np.stack(out, axis=0)                       # (K, N, 3)

    def sat_point_dirs(self, t_ns):
        """(len(t_ns), P, 3) ENU dirs of every satellite body point at each anchor time."""
        sf = self._sf_times(t_ns)
        alt, az, dist = (self.sat - self.topos).at(sf).altaz()
        ca = np.cos(alt.radians)
        pos = np.stack([dist.m * ca * np.sin(az.radians), dist.m * ca * np.cos(az.radians),
                        dist.m * np.sin(alt.radians)], axis=-1)             # (K, 3) ENU metres
        if len(pos) > 1:
            dt = (t_ns[1] - t_ns[0]) * 1e-9
            vel = np.gradient(pos, axis=0) / dt                             # (K, 3) flight direction
        else:
            vel = np.zeros_like(pos)
        earth_c = np.array([0.0, 0.0, -self.earth_r])
        nadir = earth_c - pos
        nadir /= np.linalg.norm(nadir, axis=-1, keepdims=True)
        ram = vel - (vel * nadir).sum(-1, keepdims=True) * nadir
        ram /= np.linalg.norm(ram, axis=-1, keepdims=True)
        port = np.cross(nadir, ram)
        rot = np.stack([ram, port, nadir], axis=-1)                        # (K, 3, 3) cols = body axes
        world = pos[:, None, :] + np.einsum('kij,pj->kpi', rot, self.body_pts)   # (K, P, 3)
        return world / np.linalg.norm(world, axis=-1, keepdims=True)

    def emit_group(self, writer, ids, mags, dirs, t_ns):
        """Write one record per point-target, carrying this chunk of anchors. ``dirs`` is (K, N, 3)."""
        for i, tid in enumerate(ids):
            mag = mags if np.isscalar(mags) else mags[i]
            writer.append(anchor_record(tid, mag, t_ns, dirs[:, i, :]))


def run(argv=None):
    p = argparse.ArgumentParser(description="Single sky-sim: publish star + satellite ephemeris")
    p.add_argument('--out', required=True, help="ephemeris JSONL output path")
    p.add_argument('--lat', type=float, required=True)
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--elev', type=float, default=0.0)
    p.add_argument('--epoch', required=True, help="simulated UTC epoch (ISO); maps to process start")
    p.add_argument('--tle-file', default=None)
    p.add_argument('--target-mag', type=float, default=-4.0)
    p.add_argument('--mag-limit', type=float, default=7.0)
    p.add_argument('--cache-dir', default='data/skyfield_cache')
    p.add_argument('--stop-file', default=None)
    # anchor cadence / look-ahead per group (seconds); ring in ephemeris.py must span these
    p.add_argument('--star-dt', type=float, default=30.0)
    p.add_argument('--star-lead', type=float, default=90.0)
    p.add_argument('--star-chunk', type=int, default=3)
    p.add_argument('--sat-dt', type=float, default=0.2)
    p.add_argument('--sat-lead', type=float, default=2.0)
    p.add_argument('--sat-chunk', type=int, default=10)
    args = p.parse_args(argv)

    pub = SkyPublisher(args)
    writer = JsonlWriter(args.out)
    print(f"[sky_sim] {len(pub.star_ids)} stars"
          + (f" + {len(pub.sat_ids)} sat points" if pub.sat else "")
          + f", epoch {args.epoch}", flush=True)

    star_next = pub.perf0_ns - int(args.star_dt * 1e9)     # one anchor already behind 'now'
    sat_next = pub.perf0_ns - int(args.sat_dt * 1e9)
    star_dt_ns, sat_dt_ns = int(args.star_dt * 1e9), int(args.sat_dt * 1e9)
    star_lead_ns, sat_lead_ns = int(args.star_lead * 1e9), int(args.sat_lead * 1e9)

    while True:
        if args.stop_file and os.path.exists(args.stop_file):
            break
        now = time.perf_counter_ns()

        if star_next <= now + star_lead_ns:
            t_ns = np.array([star_next + k * star_dt_ns for k in range(args.star_chunk)], dtype=np.int64)
            pub.emit_group(writer, pub.star_ids, pub.star_mag, pub.star_dirs(t_ns), t_ns.tolist())
            star_next = int(t_ns[-1]) + star_dt_ns

        if pub.sat and sat_next <= now + sat_lead_ns:
            t_ns = np.array([sat_next + k * sat_dt_ns for k in range(args.sat_chunk)], dtype=np.int64)
            pub.emit_group(writer, pub.sat_ids, pub.sat_mag, pub.sat_point_dirs(t_ns), t_ns.tolist())
            sat_next = int(t_ns[-1]) + sat_dt_ns

        time.sleep(min(args.sat_dt, 0.1))

    writer.close()


if __name__ == '__main__':
    run()
