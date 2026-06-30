"""
Physically-ish-based sky simulator for the Seeker camera.

Given a (hidden, truth) tripod orientation + optics + observer location + sim epoch, and an
*encoder* pose (which in a closed-loop test comes from the backend's published estimate),
this renders the true sky -- stars (Hipparcos) and an optional satellite (TLE/SGP4) -- into
a frame: project sources to pixels (gnomonic / pinhole), substep across the exposure and sum
to get motion streaks (mount slew + sky rotation), convolve a PSF, add shot+read noise, and
quantize to the 12-bit-in-16-bit container the real cam produces.

Positions come from Skyfield (no astropy: it handles near-field topocentric satellites
cleanly). All the math is torch -- consistent with the rest of seeker, and the forward model
is differentiable (handy for solving calibration later). CPU is fine.

The "truth" here (tripod tilt, encoder offsets, exact optics) is what the backend does NOT
know -- that's what makes a closed-loop test meaningful.
"""

import dataclasses
import datetime
import math
import os
import random

import numpy as np
import torch

from astrolock.seeker import bodies


@dataclasses.dataclass
class SkySimConfig:
    # observer
    lat_deg: float = 37.51089
    lon_deg: float = -122.2719388888889
    elev_m: float = 60.0
    epoch_utc: str = '2026-03-20T04:00:00Z'   # fixed, reproducible

    # hidden truth: mount/tripod (encoder angles -> true sky)
    az_offset_deg: float = 0.0
    alt_offset_deg: float = 0.0
    zenith_pitch_deg: float = 0.0             # tripod tilt about the east axis
    zenith_roll_deg: float = 0.0              # tripod tilt about the north axis

    # optics / sensor (hidden truth)
    width: int = 1920                         # sensor pixels (FoV = width * pixel_pitch / focal)
    height: int = 1080
    focal_length_mm: float = 8.0              # wide guide lens (~27 deg across at these defaults)
    pixel_pitch_um: float = 2.0
    roll_deg: float = 0.0                      # camera rotation about the optical axis
    boresight_px: tuple = None                 # (x, y); default = image center

    # rendering
    psf_sigma_px: float = 1.3
    star_recompute_s: float = 1.0             # recompute star apparent positions at most this
                                              # often (sim seconds); the satellite is per-substep
    mag_limit: float = 7.0
    mag_flux_scale: float = 4.0e6             # electrons/s for a mag-0 source (brighter stars)
    sky_bg_rate_e: float = 80.0              # sky background electrons / pixel / s
    read_noise_e: float = 2.0
    adu_per_e: float = 0.05                   # gain: electrons -> 12-bit ADU

    # optional satellite target: (tle_line1, tle_line2, name), plus a magnitude
    target_tle: tuple = None
    target_mag: float = 1.0
    sat_window_s: float = 600.0               # precompute the satellite ephemeris over this span
    sat_sample_s: float = 0.1                  # at this step, then interpolate (no per-frame SGP4)


def _enu(az, alt):
    """Unit vector(s) in East-North-Up from azimuth (from north, toward east) and altitude."""
    ca = torch.cos(alt)
    return torch.stack([ca * torch.sin(az), ca * torch.cos(az), torch.sin(alt)], dim=-1)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _next_fast_len(n):
    """Smallest m >= n whose only prime factors are 2,3,5,7 -- a size the FFT is fast at. Padding
    H+2*rad up to one of these avoids the slow large-prime-factor transforms (e.g. 2168 -> 2187)."""
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def ensure_cache(cache_dir='data/skyfield_cache'):
    """Download the skyfield ephemeris + Hipparcos catalog into the cache if missing, serially.
    Call this once (e.g. from the orchestrator) before launching multiple SkySim processes: they
    share one cache and would otherwise race the first-time download and clobber each other's
    rename. A no-op once the files are present."""
    from skyfield.api import Loader
    from skyfield.data import hipparcos
    os.makedirs(cache_dir, exist_ok=True)
    loader = Loader(cache_dir)
    loader.timescale()
    loader('de421.bsp')
    with loader.open(hipparcos.URL):
        pass


class SkySim:
    def __init__(self, config=None, device='cpu', cache_dir='data/skyfield_cache'):
        self.cfg = config or SkySimConfig()
        self.device = device

        from skyfield.api import Loader, wgs84, EarthSatellite
        from skyfield.data import hipparcos
        from skyfield.starlib import Star

        os.makedirs(cache_dir, exist_ok=True)
        loader = Loader(cache_dir)
        self.ts = loader.timescale()
        eph = loader('de421.bsp')                 # small ephemeris; fine for directions
        self.earth = eph['earth']
        self._topos = wgs84.latlon(self.cfg.lat_deg, self.cfg.lon_deg, elevation_m=self.cfg.elev_m)
        self.observer = self.earth + self._topos

        with loader.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)
        df = df[df['magnitude'] <= self.cfg.mag_limit]
        df = df[df['ra_degrees'].notnull() & df['dec_degrees'].notnull()]
        self.stars = Star.from_dataframe(df)
        self.star_mag = torch.tensor(df['magnitude'].to_numpy(), dtype=torch.float32, device=device)

        self.satellite = None
        if self.cfg.target_tle:
            l1, l2, name = self.cfg.target_tle
            self.satellite = EarthSatellite(l1, l2, name, self.ts)

        self._t0 = datetime.datetime.fromisoformat(self.cfg.epoch_utc.replace('Z', '+00:00'))

        c = self.cfg
        self.f_px = c.focal_length_mm / (c.pixel_pitch_um * 1e-3)   # pixels per radian-ish
        self.cx, self.cy = (c.boresight_px if c.boresight_px
                            else (c.width / 2.0, c.height / 2.0))
        # hidden tripod rotation: mount-frame ENU -> true ENU
        self._R_tilt = (_rot_x(math.radians(c.zenith_pitch_deg))
                        @ _rot_y(math.radians(c.zenith_roll_deg))).to(device)

        self._star_interval = c.star_recompute_s   # cached-star recompute cadence (+ jitter)
        self._star_cache = None
        self._next_star_t = 0.0
        self._otf = None                           # PSF transfer function, built lazily (see _psf)
        # FoV culling radius: half-diagonal of the frame + a margin for slew during the exposure
        half_x = math.atan(c.width * c.pixel_pitch_um * 1e-3 / (2 * c.focal_length_mm))
        half_y = math.atan(c.height * c.pixel_pitch_um * 1e-3 / (2 * c.focal_length_mm))
        self._cull_cos = math.cos(math.hypot(half_x, half_y) + math.radians(3.0))

        # Precompute the satellite ephemeris once (one vectorised SGP4 pass) and interpolate it
        # at render time -- no per-frame propagation. The satellite is drawn as an extended body
        # (a point-cloud model chosen by TLE name) in its LVLH attitude, so it resolves into a shape.
        self._sat_table = None
        self._body_pts = (self._t(bodies.points_for_name(c.target_tle[2]))
                          if self.satellite is not None else None)
        self._earth_r = 6371000.0 + c.elev_m       # observer geocentric radius ~ for the nadir dir
        if self.satellite is not None:
            self._sat_table = self._precompute_satellite(c.sat_window_s, c.sat_sample_s)

    def _t(self, v):
        return torch.as_tensor(v, dtype=torch.float32, device=self.device)

    # --- geometry -----------------------------------------------------------

    def boresight_basis(self, enc_az_rad, enc_alt_rad):
        """
        True boresight unit vector b and camera tangent axes A (image right = the fixed alt
        rotation axis) and L (image up = d boresight / d alt). Both come from the *mount-frame*
        orientation rotated by the tripod tilt, so they stay continuous and correct past the
        zenith (alt > 90: the tube tips over, L rolls past horizontal). For alt < 90 with no
        tilt this is identical to the simple altaz basis.
        """
        c = self.cfg
        az_m = self._t(enc_az_rad - math.radians(c.az_offset_deg))
        alt_m = self._t(enc_alt_rad - math.radians(c.alt_offset_deg))
        z = torch.zeros_like(az_m)
        # Shape-generic: scalar enc_* -> (3,) vectors; an (S,) batch of poses -> (S, 3). Rotating
        # each row vector by the tilt is `v @ R^T` (== R @ v for a single vector), so one call
        # builds every substep's basis at once.
        rt = self._R_tilt.T
        b = _enu(az_m, alt_m) @ rt
        A = torch.stack([torch.cos(az_m), -torch.sin(az_m), z], dim=-1) @ rt
        L = torch.stack([-torch.sin(alt_m) * torch.sin(az_m),
                         -torch.sin(alt_m) * torch.cos(az_m), torch.cos(alt_m)], dim=-1) @ rt
        return b, A, L

    def project(self, alt, az, b, A, L):
        """Gnomonic-project source (alt,az tensors) to pixel (x,y); returns (px, py, visible)."""
        s = _enu(az, alt)                                  # (N, 3)
        denom = s @ b
        X = (s @ A) / denom
        Y = (s @ L) / denom
        phi = math.radians(self.cfg.roll_deg)
        cphi, sphi = math.cos(phi), math.sin(phi)
        Xr = X * cphi + Y * sphi
        Yr = -X * sphi + Y * cphi
        px = self.cx + self.f_px * Xr
        py = self.cy - self.f_px * Yr
        return px, py, denom > 0

    # --- sources ------------------------------------------------------------

    def _sf_time(self, seconds_from_epoch):
        return self.ts.from_datetime(self._t0 + datetime.timedelta(seconds=seconds_from_epoch))

    def sources_altaz(self, t_sf):
        """Apparent (alt, az) radians (torch) for all stars (+ satellite), plus magnitudes."""
        alt, az, _ = self.observer.at(t_sf).observe(self.stars).apparent().altaz()
        alt = self._t(alt.radians)
        az = self._t(az.radians)
        mag = self.star_mag
        if self.satellite is not None:
            sat_alt, sat_az, _ = (self.satellite - self._topos).at(t_sf).altaz()
            alt = torch.cat([alt, self._t([sat_alt.radians])])
            az = torch.cat([az, self._t([sat_az.radians])])
            mag = torch.cat([mag, self._t([self.cfg.target_mag])])
        return alt, az, mag

    def _stars_altaz(self, t_s):
        """
        Cached star apparent (alt, az) radians. Recomputed at most every star_recompute_s sim
        seconds (the dominant cost -- Skyfield over thousands of stars); stars move only at the
        sidereal rate so this is harmless. Jittered so multiple cameras don't recompute in
        lockstep.
        """
        if self._star_cache is None or t_s >= self._next_star_t:
            alt, az, _ = self.observer.at(self._sf_time(t_s)).observe(self.stars).apparent().altaz()
            a, z = self._t(alt.radians), self._t(az.radians)
            self._star_cache = (a, z, _enu(z, a))      # cache ENU dirs too, for FoV culling
            self._next_star_t = t_s + self._star_interval * random.uniform(0.75, 1.25)
        return self._star_cache

    def _precompute_satellite(self, window_s, dt_s):
        """One vectorised SGP4 pass over [epoch, epoch+window] at dt_s. Stores the topocentric ENU
        *position* (metres) of the satellite per sample -> (dt_s, n, enu[n,3]). Cartesian ENU
        interpolates cleanly (no az wrap) and carries range, so we can recover the velocity (for the
        flight axis) by differencing and place the body points around it."""
        n = int(round(window_s / dt_s)) + 1
        secs = np.arange(n) * dt_s
        t0 = self._t0
        sec0 = t0.second + t0.microsecond * 1e-6
        times = self.ts.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute, sec0 + secs)
        alt, az, dist = (self.satellite - self._topos).at(times).altaz()
        altr, azr, dm = alt.radians, az.radians, dist.m
        ca = np.cos(altr)
        enu = np.stack([dm * ca * np.sin(azr), dm * ca * np.cos(azr), dm * np.sin(altr)], axis=-1)
        return dt_s, n, self._t(enu)

    def _sat_state_at(self, seconds):
        """Interpolate the satellite track at the given sim-times -> (pos_enu, vel_enu), each (S,3)
        metres / (m/s), from which the LVLH body frame and the body-point directions are built."""
        dt_s, n, enu = self._sat_table
        idx = (self._t(seconds) / dt_s).clamp(0, n - 1 - 1e-3)
        i0 = idx.floor().long()
        f = (idx - i0).unsqueeze(-1)
        pos = enu[i0] * (1 - f) + enu[i0 + 1] * f
        vel = (enu[i0 + 1] - enu[i0]) / dt_s        # segment velocity (the flight direction)
        return pos, vel

    # --- rendering ----------------------------------------------------------

    def _splat(self, fb, px, py, flux):
        """Bilinear-accumulate point fluxes into the framebuffer (torch)."""
        h, w = fb.shape
        x0 = torch.floor(px).long()
        y0 = torch.floor(py).long()
        fx = px - x0
        fy = py - y0
        flat = fb.view(-1)
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                            (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi = x0 + dx
            yi = y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if not bool(ok.any()):
                continue
            flat.index_add_(0, (yi[ok] * w + xi[ok]), flux[ok] * wgt[ok])

    def _psf_kernel(self):
        """The PSF as a small 2D image (sum 1), centred. Gaussian for now; a measured/aberrated
        (non-separable) PSF drops in right here -- the FFT path below convolves any kernel at the
        same cost, so a real PSF is a strictly bigger win over spatial convolution."""
        sigma = self.cfg.psf_sigma_px
        rad = max(1, int(round(3 * sigma)))
        ax = torch.arange(-rad, rad + 1, dtype=torch.float32, device=self.device)
        k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k1 /= k1.sum()
        return torch.outer(k1, k1), rad

    def _build_otf(self, h, w):
        """Precompute the PSF transfer function (FFT of the zero-padded kernel) for an (h, w) frame.
        Constant per camera, so this runs once; render-time PSF is then one rfft2 + multiply + irfft2."""
        kern, rad = self._psf_kernel()
        sz = (_next_fast_len(h + 2 * rad), _next_fast_len(w + 2 * rad))   # >= linear-conv length, fast size
        self._otf = torch.fft.rfft2(kern, s=sz)
        self._otf_sz, self._otf_rad, self._otf_hw = sz, rad, (h, w)

    def _psf(self, fb):
        # Convolve via FFT with the precomputed OTF: cost is ~independent of kernel size (unlike
        # spatial/separable conv), so it's faster even for the small Gaussian and a big win for a
        # large or non-separable (real) PSF. Zero-pad to a fast size to dodge slow prime-factor FFTs;
        # crop the centred 'same' region. Output matches the spatial conv to ~1e-6.
        h, w = fb.shape
        if self._otf is None or self._otf_hw != (h, w):
            self._build_otf(h, w)
        out = torch.fft.irfft2(torch.fft.rfft2(fb, s=self._otf_sz) * self._otf, s=self._otf_sz)
        r = self._otf_rad
        return out[r:r + h, r:r + w]

    def render(self, seconds_from_epoch, enc_az_rad, enc_alt_rad,
               rate_az_rad_s=0.0, rate_alt_rad_s=0.0, exposure_s=0.05, substeps=6):
        """Render one frame -> uint16 (H, W) ndarray (12-bit data left-shifted into 16 bits)."""
        c = self.cfg
        star_alt, star_az, star_s = self._stars_altaz(seconds_from_epoch + 0.5 * exposure_s)
        # Cull stars to the field of view (thousands -> the few dozen actually in frame): keep
        # those within the FoV diagonal (+margin) of the frame-centre boresight.
        b0, _, _ = self.boresight_basis(enc_az_rad + rate_az_rad_s * 0.5 * exposure_s,
                                        enc_alt_rad + rate_alt_rad_s * 0.5 * exposure_s)
        keep = (star_s @ b0) > self._cull_cos
        c_s, c_mag = star_s[keep], self.star_mag[keep]       # culled star ENU dirs (N,3) + mags (N,)

        # All substeps at once -- no Python loop. Build the (S,3) camera bases for every substep,
        # project the static (within-exposure) culled stars against all of them in one matmul
        # (S,N), append the satellite (one interpolated dir per substep, paired with that substep's
        # basis), then drizzle every (substep, source) point in a single splat. The exposure streak
        # is the boresight sweeping across substeps; stars don't move appreciably within a frame.
        s = torch.arange(substeps, dtype=torch.float32, device=self.device)
        fr = (s + 0.5) / substeps                            # (S,) substep mid-fractions
        az_s = self._t(enc_az_rad) + self._t(rate_az_rad_s) * fr * exposure_s
        alt_s = self._t(enc_alt_rad) + self._t(rate_alt_rad_s) * fr * exposure_s
        b, A, L = self.boresight_basis(az_s, alt_s)          # each (S,3)

        denom = b @ c_s.T                                    # (S,N): each substep basis . each star
        X = (A @ c_s.T) / denom
        Y = (L @ c_s.T) / denom
        mag = c_mag
        if self.satellite is not None:
            # Satellite as an extended body: place iss_body_points in its LVLH attitude per substep
            # (nadir toward Earth, ram along velocity, port = nadir x ram = the truss), offset from
            # the interpolated centre, then project each point's direction. So it resolves into the
            # ISS shape in a long cam, sums back to a point in the wide guide, and streaks with motion.
            pos, vel = self._sat_state_at(seconds_from_epoch + fr * exposure_s)   # (S,3),(S,3) ENU m
            earth_c = self._t([0.0, 0.0, -self._earth_r])    # observer at the ENU origin; Earth below
            nadir = earth_c - pos
            nadir = nadir / nadir.norm(dim=-1, keepdim=True)
            ram = vel - (vel * nadir).sum(-1, keepdim=True) * nadir
            ram = ram / ram.norm(dim=-1, keepdim=True)
            port = torch.cross(nadir, ram, dim=-1)           # (S,3); right-handed [ram, port, nadir]
            rot = torch.stack([ram, port, nadir], dim=-1)    # (S,3,3) columns = body axes in ENU
            world = pos[:, None, :] + torch.einsum('sij,pj->spi', rot, self._body_pts)  # (S,P,3)
            d = world / world.norm(dim=-1, keepdim=True)     # (S,P,3) unit directions
            dpt = torch.einsum('spj,sj->sp', d, b)           # (S,P) each point . its substep basis
            X = torch.cat([X, torch.einsum('spj,sj->sp', d, A) / dpt], dim=1)   # (S, N+P)
            Y = torch.cat([Y, torch.einsum('spj,sj->sp', d, L) / dpt], dim=1)
            denom = torch.cat([denom, dpt], dim=1)
            npts = self._body_pts.shape[0]                   # split the target flux over the points
            body_mag = c.target_mag + 2.5 * math.log10(npts)  # so the integrated brightness matches
            mag = torch.cat([c_mag, torch.full((npts,), body_mag, device=self.device)])

        phi = math.radians(c.roll_deg)
        cphi, sphi = math.cos(phi), math.sin(phi)
        px = self.cx + self.f_px * (X * cphi + Y * sphi)     # (S, M)
        py = self.cy - self.f_px * (-X * sphi + Y * cphi)
        vis = denom > 0
        flux = c.mag_flux_scale * (10.0 ** (-0.4 * mag)) * exposure_s / substeps   # (M,)
        flux = flux[None, :].expand(px.shape[0], -1)         # (S, M), same per substep

        fb = torch.zeros((c.height, c.width), dtype=torch.float32, device=self.device)
        self._splat(fb, px[vis], py[vis], flux[vis])         # one drizzle over all substeps

        fb = self._psf(fb)
        fb = torch.clamp(fb + c.sky_bg_rate_e * exposure_s, min=0.0)   # signal + sky bg (electrons)
        # Shot noise ~ Normal(lambda, sqrt(lambda)): a fast Gaussian approximation of Poisson
        # (exact at these electron counts, ~20x faster than torch.poisson, which dominated the
        # render), with read noise added in quadrature.
        fb = fb + torch.randn_like(fb) * torch.sqrt(fb + c.read_noise_e ** 2)
        adu = torch.clamp(torch.round(fb * c.adu_per_e), 0, 4095).to(torch.int32)
        val16 = (adu << 4)                                             # 12-bit -> 0xfff0 container
        return val16.cpu().numpy().astype(np.uint16)
