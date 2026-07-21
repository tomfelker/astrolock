"""
Single sky-simulation process: the one owner of "what is where in the sky, when".

It holds the mapping from *system time* (perf_counter_ns, shared across processes) to the *simulated*
UTC of the test pass -- chosen once, here, so nobody downstream ever sees simulated time or picks
their own epoch. It propagates every point source (Hipparcos stars via Skyfield, the target satellite
via SGP4, resolved into its body-point cloud in an LVLH attitude) and publishes their topocentric ENU
directions as piecewise-linear anchors, in system time, to a JSONL almanac (see almanac.py).

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
from astrolock.seeker.almanac import fix_record
from astrolock.seeker.sidecar import JsonlWriter
from astrolock.seeker.session import mono_ns, parent_lifeline
from astrolock.seeker.tles import tle_epoch_utc

# Solar-system bodies for the NAVIGATION feed (alignment overlay): point targets like
# everything else, plus a display name and a true angular radius (the GUI draws the disc at
# scale). de421 keys tried in order (it has the inner planets proper, barycenters for the
# rest -- at overlay precision the difference is arcseconds). Magnitudes are rough visual
# constants; good enough for a glyph.
NAV_BODIES = [   # (target id, display name, de421 keys, visual mag, body radius km)
    ('sun', 'Sun', ('sun',), -26.7, 696000.0),
    ('moon', 'Moon', ('moon',), -12.7, 1737.4),
    ('planet:mercury', 'Mercury', ('mercury',), 0.2, 2439.7),
    ('planet:venus', 'Venus', ('venus',), -4.1, 6051.8),
    ('planet:mars', 'Mars', ('mars', 'mars barycenter'), -0.5, 3389.5),
    ('planet:jupiter', 'Jupiter', ('jupiter barycenter',), -2.2, 69911.0),
    ('planet:saturn', 'Saturn', ('saturn barycenter',), 0.6, 58232.0),
    ('planet:uranus', 'Uranus', ('uranus barycenter',), 5.7, 25362.0),
    ('planet:neptune', 'Neptune', ('neptune barycenter',), 7.8, 24622.0),
]


def _star_names(loader):
    """HIP -> proper name from the IAU Working Group on Star Names catalog. The skyfield
    loader caches the download; offline with a cold cache -> {} and labels fall back to
    HIP ids. Fixed-column text format (the old astrolock parser, condensed)."""
    url = 'https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt'
    try:
        with loader.open(url) as f:
            text = f.read().decode('utf-8')
    except Exception as e:
        print(f"[sky_sim] star-name catalog unavailable ({e}); labels fall back to HIP ids",
              flush=True)
        return {}
    out = {}
    for line in text.split('\n'):
        if not line or line[0] in ('#', '$'):
            continue
        hip = line[90:96].strip() if len(line) >= 96 else ''
        if hip.isdigit():
            out[int(hip)] = line[18:36].rstrip()
    return out


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
        # Navigation subset: the few brightest stars, published (much less often) to the
        # sparse navigation feed the GUI overlays -- alignment needs Vega, not 15k stars.
        nav_df = df[df['magnitude'] <= args.nav_mag]
        self.nav_stars = Star.from_dataframe(nav_df)
        self.nav_ids = [f"star:{int(h)}" for h in nav_df.index.to_numpy()]
        self.nav_mag = nav_df['magnitude'].to_numpy().astype(float)
        proper = _star_names(loader)
        self.nav_names = [proper.get(int(h)) for h in nav_df.index.to_numpy()]
        # Sun/Moon/planets for the nav feed: same point-target machinery, from the ephemeris
        # already loaded. Missing keys are skipped (ephemeris variants differ).
        self.nav_bodies = []
        for tid, disp, keys, mag, radius_km in NAV_BODIES:
            for k in keys:
                try:
                    self.nav_bodies.append((tid, disp, eph[k], mag, radius_km))
                    break
                except (KeyError, ValueError):
                    pass

        self.sat = None
        if args.tle_file:
            name, l1, l2 = _load_tle(args.tle_file)
            self.sat = EarthSatellite(l1, l2, name, self.ts)
            self.sat_name = name
            self.sat_epoch_utc = tle_epoch_utc(l1)
            self.body_pts = bodies.points_for_name(name).astype(float)     # (P, 3) body-frame metres
            npts = len(self.body_pts)
            self.sat_ids = [f"{name} point {p}/{npts}" for p in range(npts)]
            self.sat_mag = args.target_mag + 2.5 * math.log10(npts)        # split flux over the points

        self.epoch = datetime.datetime.fromisoformat(args.epoch.replace('Z', '+00:00'))
        self.perf0_ns = mono_ns()                                          # process start (emit cursors)
        # The system-time <-> UTC anchor: sim UTC == --epoch at system time --epoch-t-ns. The
        # backend passes an explicit anchor so time is absolute on the shared timeline (a respawn
        # at the same anchor+epoch continues the SAME time); 0 = anchor to process start (standalone).
        self.epoch_t_ns = args.epoch_t_ns or self.perf0_ns

    def _sf_secs(self, t_ns):
        """Seconds-from-epoch (as skyfield.ts.utc wants them) for system-time anchor(s) t_ns."""
        return (np.asarray(t_ns, dtype=np.float64) - self.epoch_t_ns) * 1e-9

    def _sf_times(self, t_ns):
        """Skyfield Time for system-time anchors t_ns (scalar -> single Time, array -> Time array)."""
        e = self.epoch
        return self.ts.utc(e.year, e.month, e.day, e.hour, e.minute,
                           e.second + e.microsecond * 1e-6 + self._sf_secs(t_ns))

    def star_dirs(self, t_ns, stars=None):
        """(len(t_ns), N_stars, 3) ENU dirs of every star at each anchor time. Uses a *scalar* time
        per anchor -- observing N stars against an array time mis-broadcasts in Skyfield.
        ``stars`` defaults to the full catalog; pass ``self.nav_stars`` for the sparse subset."""
        out = []
        for t in t_ns:                                    # a few anchors per emit; catalog pass each
            alt, az, _ = (self.observer.at(self._sf_times(float(t)))
                          .observe(stars if stars is not None else self.stars).apparent().altaz())
            out.append(_enu_from_altaz(np.asarray(az.radians).reshape(-1),
                                       np.asarray(alt.radians).reshape(-1)))
        return np.stack(out, axis=0)                       # (K, N, 3)

    def sat_center_dirs(self, t_ns):
        """(len(t_ns), 3) ENU unit dirs of the satellite's CENTRE -- the navigation feed's pass
        track (one target, coarse anchors), vs the dense per-body-point group above."""
        sf = self._sf_times(np.asarray(t_ns, dtype=np.int64))
        alt, az, _ = (self.sat - self.topos).at(sf).altaz()
        return _enu_from_altaz(np.asarray(az.radians).reshape(-1),
                               np.asarray(alt.radians).reshape(-1))

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

    def body_fixes(self, obj, t_ns):
        """((K, 3) ENU dirs, distance_km at the last anchor) for one ephemeris body."""
        dirs = []
        dist_km = 0.0
        for t in t_ns:
            alt, az, dist = (self.observer.at(self._sf_times(float(t)))
                             .observe(obj).apparent().altaz())
            dirs.append(_enu_from_altaz(float(az.radians), float(alt.radians)))
            dist_km = float(dist.km)
        return np.stack(dirs, axis=0), dist_km

    def emit_group(self, writer, ids, mags, dirs, t_ns, names=None):
        """Write one record per point-target, carrying this chunk of fixes. ``dirs`` is (K, N, 3).
        ``names`` (optional, aligned with ids): display names; None entries are omitted."""
        for i, tid in enumerate(ids):
            mag = mags if np.isscalar(mags) else mags[i]
            writer.append(fix_record(tid, mag, t_ns, dirs[:, i, :],
                                     name=names[i] if names else None))


def run(argv=None):
    p = argparse.ArgumentParser(description="Single sky-sim: publish star + satellite almanac")
    p.add_argument('--out', required=True, help="almanac JSONL output path")
    p.add_argument('--lat', type=float, required=True)
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--elev', type=float, default=0.0)
    p.add_argument('--epoch', required=True, help="UTC epoch (ISO) at system time --epoch-t-ns")
    p.add_argument('--epoch-t-ns', type=int, default=0,
                   help="system time (mono_ns) at which sim UTC == --epoch (0 = process start)")
    p.add_argument('--tle-file', default=None)
    p.add_argument('--target-mag', type=float, default=-4.0)
    p.add_argument('--mag-limit', type=float, default=7.0)
    p.add_argument('--cache-dir', default='data/skyfield_cache')
    p.add_argument('--stop-file', default=None)
    # fix cadence / look-ahead per group (seconds). The almanac evicts by age (floor-and-newer), so
    # its buffer adapts to these -- there's no ring size to keep in agreement across processes.
    p.add_argument('--star-dt', type=float, default=30.0)
    p.add_argument('--star-lead', type=float, default=60.0)
    p.add_argument('--star-chunk', type=int, default=2)
    p.add_argument('--star-write-batch', type=int, default=2000,
                   help="stars written per loop iteration -- dribble the ~15k catalog out (one target "
                        "per line) so no consumer parses it all in a single poll")
    p.add_argument('--sat-dt', type=float, default=0.2)
    p.add_argument('--sat-lead', type=float, default=2.0)
    p.add_argument('--sat-chunk', type=int, default=10)
    # Sparse NAVIGATION feed (GUI alignment overlay): a handful of bright stars + the
    # satellite's centre-of-pass track, at leisurely cadence -- kilobytes/minute.
    p.add_argument('--nav-out', default=None, help="navigation JSONL output path (None = off)")
    p.add_argument('--nav-mag', type=float, default=2.5,
                   help="navigation feed: include stars at least this bright (~50 stars)")
    p.add_argument('--nav-dt', type=float, default=15.0, help="navigation star anchor spacing (s)")
    p.add_argument('--nav-lead', type=float, default=60.0, help="navigation publish look-ahead (s)")
    p.add_argument('--nav-sat-dt', type=float, default=5.0,
                   help="navigation satellite-track anchor spacing (s)")
    p.add_argument('--nav-sat-horizon', type=float, default=600.0,
                   help="how far ahead the satellite pass track is published (s)")
    args = p.parse_args(argv)

    pub = SkyPublisher(args)
    writer = JsonlWriter(args.out)
    # Streams are append-only across respawns (a time/site change restarts this process into the
    # SAME files). Everything already published -- including fixes minutes into the future -- is
    # wrong under the new time/site, so open with a reset: consumers drop all targets on sight.
    writer.append({'reset': True})
    tle_note = ""
    if pub.sat:
        age_d = (pub.epoch - pub.sat_epoch_utc).total_seconds() / 86400.0
        tle_note = (f" [{pub.sat_name} TLE epoch {pub.sat_epoch_utc:%Y-%m-%d}, "
                    f"{abs(age_d):.1f} d {'older' if age_d >= 0 else 'newer'} than sim time]")
    print(f"[sky_sim] {len(pub.star_ids)} stars"
          + (f" + {len(pub.sat_ids)} sat points" if pub.sat else "")
          + (f", nav feed {len(pub.nav_ids)} stars + {len(pub.nav_bodies)} bodies"
             f" -> {args.nav_out}" if args.nav_out else "")
          + f", epoch {args.epoch}" + tle_note, flush=True)

    star_next = pub.perf0_ns - int(args.star_dt * 1e9)     # one anchor already behind 'now'
    sat_next = pub.perf0_ns - int(args.sat_dt * 1e9)
    star_dt_ns, sat_dt_ns = int(args.star_dt * 1e9), int(args.sat_dt * 1e9)
    star_lead_ns, sat_lead_ns = int(args.star_lead * 1e9), int(args.sat_lead * 1e9)
    nstars = len(pub.star_ids)
    star_pending = None                                    # [t_ns list, dirs (K,N,3), cursor]

    nav_writer = JsonlWriter(args.nav_out) if args.nav_out else None
    if nav_writer is not None:
        nav_writer.append({'reset': True})                 # same respawn contract as the main stream
    nav_next = pub.perf0_ns - int(args.nav_dt * 1e9)       # nav stars: one anchor behind 'now'
    nav_track_next = pub.perf0_ns                          # sat track: forward-only cursor (no
    nav_dt_ns = int(args.nav_dt * 1e9)                     # overlapping re-emits -- fixes append)
    nav_lead_ns = int(args.nav_lead * 1e9)
    nav_sat_dt_ns = int(args.nav_sat_dt * 1e9)
    nav_horizon_ns = int(args.nav_sat_horizon * 1e9)

    parent_dead = parent_lifeline()                        # backend gone (however it died) -> stop
    while True:
        if parent_dead.is_set() or (args.stop_file and os.path.exists(args.stop_file)):
            break
        now = mono_ns()

        # Satellite: dense but low-volume (~68 points) -- write its whole chunk when due.
        if pub.sat and sat_next <= now + sat_lead_ns:
            t_ns = np.array([sat_next + k * sat_dt_ns for k in range(args.sat_chunk)], dtype=np.int64)
            pub.emit_group(writer, pub.sat_ids, pub.sat_mag, pub.sat_point_dirs(t_ns), t_ns.tolist())
            sat_next = int(t_ns[-1]) + sat_dt_ns

        # Stars: compute the next anchor chunk when due, then dribble the ~15k targets out a batch at
        # a time so no single poll re-parses the whole catalog.
        if star_pending is None and star_next <= now + star_lead_ns:
            t_ns = [int(star_next + k * star_dt_ns) for k in range(args.star_chunk)]
            star_pending = [t_ns, pub.star_dirs(np.array(t_ns, dtype=np.int64)), 0]
            star_next = t_ns[-1] + star_dt_ns
        if star_pending is not None:
            t_ns, sd, cur = star_pending
            end = min(cur + args.star_write_batch, nstars)
            pub.emit_group(writer, pub.star_ids[cur:end], pub.star_mag[cur:end], sd[:, cur:end], t_ns)
            star_pending = None if end >= nstars else [t_ns, sd, end]

        # Navigation feed: ~50 bright stars every nav_dt, plus the satellite pass track
        # published forward-only out to the horizon (fixes only ever append; no overlaps).
        if nav_writer is not None and nav_next <= now + nav_lead_ns:
            t_ns = [int(nav_next), int(nav_next + nav_dt_ns)]
            pub.emit_group(nav_writer, pub.nav_ids, pub.nav_mag,
                           pub.star_dirs(np.array(t_ns, dtype=np.int64), stars=pub.nav_stars),
                           t_ns, names=pub.nav_names)
            for tid, disp, obj, bmag, radius_km in pub.nav_bodies:   # sun/moon/planets
                bd, dist_km = pub.body_fixes(obj, t_ns)
                nav_writer.append(fix_record(
                    tid, bmag, t_ns, bd, name=disp,
                    angular_radius_rad=math.asin(min(1.0, radius_km / max(dist_km, radius_km)))))
            nav_next = t_ns[-1] + nav_dt_ns
        if nav_writer is not None and pub.sat and nav_track_next <= now + nav_horizon_ns:
            t_ns = np.arange(nav_track_next, now + nav_horizon_ns, nav_sat_dt_ns, dtype=np.int64)
            if len(t_ns):
                nav_writer.append(fix_record('sat:track', args.target_mag, t_ns.tolist(),
                                             pub.sat_center_dirs(t_ns), name=pub.sat_name))
                nav_track_next = int(t_ns[-1]) + nav_sat_dt_ns

        time.sleep(min(args.sat_dt, 0.05))

    writer.close()
    if nav_writer is not None:
        nav_writer.close()


if __name__ == '__main__':
    run()
