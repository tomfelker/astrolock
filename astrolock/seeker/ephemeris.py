"""
Shared sky ephemeris. One process (sky_sim) propagates every point source -- stars and satellite
body points alike -- and publishes their topocentric ENU directions as piecewise-linear anchors in
*system time* (perf_counter_ns, shared across processes). Consumers (the camera sims now, the GUI
later once we have plate-solving) read this file and interpolate.

Uniform model: a "target" is a point source with an id, a magnitude, and a stream of
(t_mono_ns, unit_dir) anchors. Stars and satellite points differ only in anchor *density* -- stars
move at the sidereal rate (minutes between anchors), a fast satellite needs a fraction of a second.
Directions (not az/alt) keep it coordinate-free and wrap-free, and put the whole lookup in one
batched torch lerp over (target, anchor) tensors.

Why this exists: previously each camera process propagated the sky itself, timed against its *own*
process start -- so two cameras placed a fast satellite at slightly different world-times and drifted
apart. Centralising propagation (one clock, one SGP4/catalog pass) fixes that, kills the once-a-second
star recompute jump, and lets the GUI reuse the exact same positions.

The dense (t, dir) tables are maintained *incrementally*: new anchors refresh only the rows that
changed (a sat emit touches ~68 rows, not the whole ~15k-star table), so per-frame lookup stays cheap.

File format: JSONL, one record per target-extension::

    {"id": "<str>", "mag": <float>, "anchors": [[t_mono_ns, x, y, z], ...]}
"""

from collections import defaultdict

import numpy as np
import torch

from astrolock.seeker.sidecar import JsonlTailer


def anchor_record(target_id, mag, times_ns, dirs):
    """Build a JSONL record extending ``target_id`` with anchors ``(times_ns[i], dirs[i])``.
    ``dirs`` is an (n, 3) array-like of unit ENU directions."""
    return {'id': target_id, 'mag': float(mag),
            'anchors': [[int(t), float(d[0]), float(d[1]), float(d[2])]
                        for t, d in zip(times_ns, dirs)]}


class SkyEphemeris:
    """Tails an ephemeris JSONL and answers ``dirs_at(times)`` for every target at once (torch)."""

    def __init__(self, path, ring=32, device='cpu'):
        # ring must span more than sky_sim's emit-ahead window for the densest target, or the newest
        # anchors evict the ones bracketing `now` and lookups clamp+jump (a fast satellite then
        # zig-zags). For the sat defaults (0.2 s spacing, ~4 s ahead) that needs > 20; 32 leaves margin.
        self.device = device
        self.ring = ring
        self._tailer = JsonlTailer(path)
        self._ids = []
        self._index = {}
        self._at = []           # per target: list[int ns] (sorted, <= ring)
        self._ad = []           # per target: list[[x, y, z]]
        self._mag = []
        self._pending = set()   # target indices whose tensor rows need refreshing
        self._times = torch.zeros((0, ring), dtype=torch.int64, device=device)
        self._dirs = torch.zeros((0, ring, 3), dtype=torch.float32, device=device)
        self._mag_t = torch.zeros((0,), dtype=torch.float32, device=device)

    def update(self):
        """Ingest newly-committed records. Cheap; call once per frame before dirs_at(). Only marks
        which rows changed -- the tensor refresh happens lazily in dirs_at, for those rows only."""
        new = 0
        for rec in self._tailer.poll():
            tid = rec.get('id')
            if tid is None:
                continue
            i = self._index.get(tid)
            if i is None:
                i = len(self._ids)
                self._index[tid] = i
                self._ids.append(tid)
                self._at.append([])
                self._ad.append([])
                self._mag.append(float(rec.get('mag', 12.0)))
                new += 1
            if 'mag' in rec:
                self._mag[i] = float(rec['mag'])
            at, ad = self._at[i], self._ad[i]
            for a in rec.get('anchors', ()):
                t = int(a[0])
                if at and t <= at[-1]:
                    continue                       # ignore out-of-order / duplicate anchors
                at.append(t)
                ad.append([float(a[1]), float(a[2]), float(a[3])])
            if len(at) > self.ring:                # keep only the newest `ring` anchors
                del at[:len(at) - self.ring]
                del ad[:len(ad) - self.ring]
            self._pending.add(i)
        if new:                                    # grow the tensors for the new targets
            z_t = torch.zeros((new, self.ring), dtype=torch.int64, device=self.device)
            z_d = torch.zeros((new, self.ring, 3), dtype=torch.float32, device=self.device)
            z_m = torch.zeros((new,), dtype=torch.float32, device=self.device)
            self._times = torch.cat([self._times, z_t], 0)
            self._dirs = torch.cat([self._dirs, z_d], 0)
            self._mag_t = torch.cat([self._mag_t, z_m], 0)

    def _flush(self):
        """Refresh only the changed rows. Group them by anchor-count and build each group with one
        batched np.array (no Python-per-row overhead) -- all stars in an emit share a count, so this
        stays one or two batches whether they're warming up or full. Rows shorter than `ring` are
        left-padded by repeating the oldest anchor at strictly-earlier fake times, so rows stay sorted."""
        if not self._pending:
            return
        idx = sorted(self._pending)
        self._pending.clear()
        R = self.ring
        by_n = defaultdict(list)
        for i in idx:
            by_n[len(self._at[i])].append(i)
        for n, group in by_n.items():
            tt = np.zeros((len(group), R), dtype=np.int64)
            dd = np.zeros((len(group), R, 3), dtype=np.float32)
            if n == 0:
                tt[:] = np.arange(R)
            else:
                ats = np.array([self._at[i] for i in group], dtype=np.int64)       # (Kg, n)
                ads = np.array([self._ad[i] for i in group], dtype=np.float32)     # (Kg, n, 3)
                pad = R - n
                if pad > 0:                                    # left-pad below the oldest real anchor
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
        self._flush()
        T = self._times.shape[0]
        if T == 0:
            return (torch.zeros((0, t_ns.numel(), 3), device=self.device),
                    torch.zeros((0,), device=self.device))
        times = self._times                                       # (T, R) int64
        R = times.shape[1]
        vals = t_ns.to(torch.int64).view(-1).unsqueeze(0).expand(T, -1).contiguous()   # (T, S)
        idx = torch.searchsorted(times, vals).clamp(1, R - 1)     # per-row segment upper index
        lo = idx - 1
        t0 = torch.gather(times, 1, lo).double()
        t1 = torch.gather(times, 1, idx).double()
        frac = ((vals.double() - t0) / (t1 - t0).clamp(min=1.0)).clamp(0.0, 1.0).float()  # (T, S)
        d0 = torch.gather(self._dirs, 1, lo.unsqueeze(-1).expand(-1, -1, 3))   # (T, S, 3)
        d1 = torch.gather(self._dirs, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        d = d0 + (d1 - d0) * frac.unsqueeze(-1)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return d, self._mag_t

    @property
    def ids(self):
        return list(self._ids)

    def close(self):
        self._tailer.close()
