"""
Shared sky almanac. One process (sky_sim) propagates every point source -- stars and satellite body
points alike -- and publishes their topocentric ENU directions as piecewise-linear *fixes*
(time, unit_dir) in system time (perf_counter_ns, shared across processes). Consumers (the camera
sims now, the GUI later once we have plate-solving) read this file and interpolate.

Like a nautical/astronomical almanac: a lookup table of many bodies' positions over time. A "target"
is a point source with an id, a magnitude, and a stream of fixes; stars and satellite points differ
only in fix *density* (stars: tens of seconds apart; a fast satellite: a fraction of a second).
Directions (not az/alt) keep it coordinate-free and wrap-free, so the lookup is one batched torch
lerp over (target, fix) tensors.

Design (two structures, one job each):
  - the unbounded source of truth is a per-target Python list of fixes -- it never discards a fix it
    might still need. Eviction is by *age*, not count: we keep the "floor" fix (newest with
    time <= the OLDEST time of the last query batch) and everything after it, dropping strictly older
    ones, so a re-query at that time still lands inside a real segment -- and a batch that also looks
    into the future (the GUI's satellite pass line) can't evict the present. This makes retention
    independent of how far ahead the producer publishes -- no ring size to agree on across processes.
  - a dense torch tensor is a *derived cache* for the batched lerp; its width is data-driven (the max
    fixes-in-window across targets, grown in steps), not a fixed tunable.

Timing: fixes/queries are int64 ns. The lerp differences (vals - t0) in int64 -- fixed-width
two's-complement, so a difference is correct even if perf_counter's absolute value has wrapped, and
the small result is exact -- then float32 is plenty for the fraction (sub-microsecond).

File format: JSONL, one record per target-extension:
    {"id": "<str>", "mag": <float>, "fixes": [[t_mono_ns, x, y, z], ...]}
plus {"reset": true}, written by a (re)starting publisher: drop every target -- the time or site
changed, so all previously published fixes (even future ones) are wrong.
"""

import bisect
from collections import defaultdict

import numpy as np
import torch

from astrolock.seeker.sidecar import JsonlTailer


def fix_record(target_id, mag, times_ns, dirs, name=None, angular_radius_rad=None):
    """Build a JSONL record extending ``target_id`` with fixes ``(times_ns[i], dirs[i])``.
    ``dirs`` is an (n, 3) array-like of unit ENU directions. Optional per-target metadata
    rides along: a display ``name`` (GUI labels) and an ``angular_radius_rad`` (true
    angular size -- the GUI draws sun/moon/planet discs at scale)."""
    rec = {'id': target_id, 'mag': float(mag),
           'fixes': [[int(t), float(d[0]), float(d[1]), float(d[2])]
                     for t, d in zip(times_ns, dirs)]}
    if name is not None:
        rec['name'] = str(name)
    if angular_radius_rad is not None:
        rec['angular_radius_rad'] = float(angular_radius_rad)
    return rec


def _grow_cols(need):
    """Tensor width for `need` fixes, rounded up in steps of 8 (a high-water mark, so the width
    changes rarely rather than on every fix)."""
    return max(8, ((need + 7) // 8) * 8)


class SkyAlmanac:
    """Tails an almanac JSONL and answers ``dirs_at(times)`` for every target at once (torch)."""

    def __init__(self, path, device='cpu'):
        self.device = device
        self._tailer = JsonlTailer(path)
        self._ids = []
        self._index = {}
        self._t = []            # per target: list[int ns], sorted -- unbounded source of truth
        self._d = []            # per target: list[[x, y, z]]
        self._mag = []
        self._name = []         # per target: display name (defaults to the id)
        self._angular_radius = []   # per target: true angular radius (rad), or None
        self._pending = set()   # rows whose fix list changed since the last flush
        self._last_q = None     # newest query time seen (drives age-based eviction)
        self._cols = 8          # current tensor width (data-driven high-water mark)
        self._times = torch.zeros((0, self._cols), dtype=torch.int64, device=device)
        self._dirs = torch.zeros((0, self._cols, 3), dtype=torch.float32, device=device)
        self._mag_t = torch.zeros((0,), dtype=torch.float32, device=device)

    def update(self):
        """Ingest newly-committed records; append fixes and evict stale ones (floor-and-newer)."""
        new = 0
        for rec in self._tailer.poll():
            if rec.get('reset'):
                # A respawned sky_sim (time/site changed) opens its stream with {"reset": true}:
                # every fix published before it -- including future ones, which the out-of-order
                # guard below would otherwise trust over the corrections -- is now wrong.
                self._ids, self._index, self._t, self._d, self._mag = [], {}, [], [], []
                self._name, self._angular_radius = [], []
                self._pending.clear()
                new = 0
                self._times = torch.zeros((0, self._cols), dtype=torch.int64, device=self.device)
                self._dirs = torch.zeros((0, self._cols, 3), dtype=torch.float32, device=self.device)
                self._mag_t = torch.zeros((0,), dtype=torch.float32, device=self.device)
                continue
            tid = rec.get('id')
            if tid is None:
                continue
            i = self._index.get(tid)
            if i is None:
                i = len(self._ids)
                self._index[tid] = i
                self._ids.append(tid)
                self._t.append([])
                self._d.append([])
                self._mag.append(float(rec.get('mag', 12.0)))
                self._name.append(tid)
                self._angular_radius.append(None)
                new += 1
            if 'mag' in rec:
                self._mag[i] = float(rec['mag'])
            if 'name' in rec:
                self._name[i] = str(rec['name'])
            if 'angular_radius_rad' in rec:
                self._angular_radius[i] = float(rec['angular_radius_rad'])
            ts, ds = self._t[i], self._d[i]
            for f in rec.get('fixes', ()):
                t = int(f[0])
                if ts and t <= ts[-1]:
                    continue                       # ignore out-of-order / duplicate fixes
                ts.append(t)
                ds.append([float(f[1]), float(f[2]), float(f[3])])
            if self._last_q is not None:           # keep the floor fix (<= last query) and newer
                k = bisect.bisect_right(ts, self._last_q) - 1
                if k > 0:
                    del ts[:k]
                    del ds[:k]
            self._pending.add(i)
        if new:                                    # grow the tensors for the new targets
            z_t = torch.zeros((new, self._cols), dtype=torch.int64, device=self.device)
            z_d = torch.zeros((new, self._cols, 3), dtype=torch.float32, device=self.device)
            z_m = torch.zeros((new,), dtype=torch.float32, device=self.device)
            self._times = torch.cat([self._times, z_t], 0)
            self._dirs = torch.cat([self._dirs, z_d], 0)
            self._mag_t = torch.cat([self._mag_t, z_m], 0)

    def _flush(self):
        if not self._pending:
            return
        need = max((len(t) for t in self._t), default=1)
        if need > self._cols:                      # widen (grow only) and rebuild every row
            self._cols = _grow_cols(need)
            n = len(self._ids)
            self._times = torch.zeros((n, self._cols), dtype=torch.int64, device=self.device)
            self._dirs = torch.zeros((n, self._cols, 3), dtype=torch.float32, device=self.device)
            self._mag_t = torch.zeros((n,), dtype=torch.float32, device=self.device)
            rows = range(n)
        else:
            rows = sorted(self._pending)
        self._pending.clear()
        self._build(rows)

    def _build(self, rows):
        """(Re)build the given tensor rows from the Python fix lists, grouped by fix-count so each
        group is one batched np.array (no Python-per-row overhead)."""
        R = self._cols
        by_n = defaultdict(list)
        for i in rows:
            by_n[len(self._t[i])].append(i)
        for n, group in by_n.items():
            tt = np.zeros((len(group), R), dtype=np.int64)
            dd = np.zeros((len(group), R, 3), dtype=np.float32)
            if n == 0:
                tt[:] = np.arange(R)
            else:
                ats = np.array([self._t[i] for i in group], dtype=np.int64)       # (Kg, n)
                ads = np.array([self._d[i] for i in group], dtype=np.float32)     # (Kg, n, 3)
                pad = R - n
                if pad > 0:                        # left-pad below the oldest real fix, sorted
                    tt[:, :pad] = ats[:, 0:1] - np.arange(pad, 0, -1)[None, :]
                    dd[:, :pad] = ads[:, 0:1, :]
                tt[:, pad:] = ats
                dd[:, pad:] = ads
            ii = torch.tensor(group, dtype=torch.int64, device=self.device)
            self._times[ii] = torch.from_numpy(tt).to(self.device)
            self._dirs[ii] = torch.from_numpy(dd).to(self.device)
            self._mag_t[ii] = torch.tensor([self._mag[i] for i in group],
                                           dtype=torch.float32, device=self.device)

    def dirs_at(self, t_ns):
        """Interpolated unit directions at query times ``t_ns`` (shape (S,), int64 ns).
        Returns (dirs (T, S, 3), mags (T,)). One batched piecewise-linear lerp over all targets."""
        t_ns = t_ns.to(torch.int64).view(-1)
        # Retention keys to the OLDEST query time in the batch: that's the earliest time a
        # consumer might still re-ask about. Keying to the newest was fine when consumers
        # only queried 'now', but a batch that looks into the FUTURE (the GUI's satellite
        # pass line, +10 min) would evict every current fix -- stars then clamp to stale
        # anchors and the track loses its geometry.
        self._last_q = int(t_ns.min())
        self._flush()
        T = self._times.shape[0]
        if T == 0:
            return (torch.zeros((0, t_ns.numel(), 3), device=self.device),
                    torch.zeros((0,), device=self.device))
        times = self._times                                       # (T, R) int64
        R = times.shape[1]
        vals = t_ns.unsqueeze(0).expand(T, -1).contiguous()       # (T, S) int64
        idx = torch.searchsorted(times, vals).clamp(1, R - 1)
        lo = idx - 1
        t0 = torch.gather(times, 1, lo)
        t1 = torch.gather(times, 1, idx)
        # Difference in int64 (fixed-width wraparound -> correct across a perf_counter wrap, exact),
        # then float32 is plenty for the fraction.
        num = (vals - t0).to(torch.float32)
        den = (t1 - t0).clamp(min=1).to(torch.float32)
        frac = (num / den).clamp(0.0, 1.0)
        d0 = torch.gather(self._dirs, 1, lo.unsqueeze(-1).expand(-1, -1, 3))
        d1 = torch.gather(self._dirs, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        d = d0 + (d1 - d0) * frac.unsqueeze(-1)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return d, self._mag_t

    @property
    def ids(self):
        return list(self._ids)

    @property
    def names(self):
        """Per-target display names, aligned with ``ids`` (default: the id itself)."""
        return list(self._name)

    @property
    def angular_radius_rad(self):
        """Per-target true angular radius (rad) aligned with ``ids``; None for point sources."""
        return list(self._angular_radius)

    def close(self):
        self._tailer.close()
