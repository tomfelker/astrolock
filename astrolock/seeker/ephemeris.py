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

File format: JSONL, one record per target-extension::

    {"id": "<str>", "mag": <float>, "anchors": [[t_mono_ns, x, y, z], ...]}

Anchors append to that target's ring (newest kept); a consumer lerps the two that bracket its query.
"""

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

    def __init__(self, path, ring=16, device='cpu'):
        self.device = device
        self.ring = ring
        self._tailer = JsonlTailer(path)
        self._ids = []
        self._index = {}
        self._at = []           # per target: list[int ns] (sorted, <= ring)
        self._ad = []           # per target: list[[x, y, z]]
        self._mag = []
        self._dirty = True
        self._times = None      # (T, ring) int64 ns
        self._dirs = None       # (T, ring, 3) float32
        self._mag_t = None      # (T,) float32

    def update(self):
        """Ingest any newly-committed records. Cheap; call once per frame before dirs_at()."""
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
            self._dirty = True

    def _rebuild(self):
        """Pack the ragged per-target rings into dense (T, ring) tensors. Rows shorter than `ring`
        are left-padded by repeating the oldest anchor at strictly-earlier fake times, so every row
        is sorted and any real query time lands in the real (non-padded) span."""
        T, R = len(self._ids), self.ring
        times = torch.zeros((T, R), dtype=torch.int64)
        dirs = torch.zeros((T, R, 3), dtype=torch.float32)
        for i in range(T):
            at, ad = self._at[i], self._ad[i]
            n = len(at)
            if n == 0:
                times[i] = torch.arange(R, dtype=torch.int64)     # harmless placeholder
                continue
            pad = R - n
            for k in range(R):
                if k < pad:
                    times[i, k] = at[0] - (pad - k)               # strictly increasing, all < at[0]
                    dirs[i, k] = torch.tensor(ad[0])
                else:
                    times[i, k] = at[k - pad]
                    dirs[i, k] = torch.tensor(ad[k - pad])
        self._times = times.to(self.device)
        self._dirs = dirs.to(self.device)
        self._mag_t = torch.tensor(self._mag, dtype=torch.float32, device=self.device)
        self._dirty = False

    def dirs_at(self, t_ns):
        """Interpolated unit directions at query times ``t_ns`` (shape (S,), int64 ns).
        Returns (dirs (T, S, 3), mags (T,)). One batched piecewise-linear lerp over all targets."""
        if self._dirty:
            self._rebuild()
        if not self._ids:
            return (torch.zeros((0, t_ns.numel(), 3), device=self.device),
                    torch.zeros((0,), device=self.device))
        times = self._times                                       # (T, R) int64
        T, R = times.shape
        vals = t_ns.to(torch.int64).view(-1).unsqueeze(0).expand(T, -1).contiguous()   # (T, S)
        idx = torch.searchsorted(times, vals).clamp(1, R - 1)     # per-row segment upper index
        lo = idx - 1
        t0 = torch.gather(times, 1, lo).double()
        t1 = torch.gather(times, 1, idx).double()
        frac = ((vals.double() - t0) / (t1 - t0).clamp(min=1.0)).clamp(0.0, 1.0).float()  # (T, S)
        g_lo = lo.unsqueeze(-1).expand(-1, -1, 3)
        g_hi = idx.unsqueeze(-1).expand(-1, -1, 3)
        d0 = torch.gather(self._dirs, 1, g_lo)                    # (T, S, 3)
        d1 = torch.gather(self._dirs, 1, g_hi)
        d = d0 + (d1 - d0) * frac.unsqueeze(-1)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return d, self._mag_t

    @property
    def ids(self):
        return list(self._ids)

    def close(self):
        self._tailer.close()
