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
        # FoV culling radius: half-diagonal of the frame + a margin for slew during the exposure
        half_x = math.atan(c.width * c.pixel_pitch_um * 1e-3 / (2 * c.focal_length_mm))
        half_y = math.atan(c.height * c.pixel_pitch_um * 1e-3 / (2 * c.focal_length_mm))
        self._cull_cos = math.cos(math.hypot(half_x, half_y) + math.radians(3.0))

        # Precompute the satellite ephemeris once (one vectorised SGP4 pass) and interpolate it
        # at render time -- no per-frame propagation.
        self._sat_table = None
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
        z = torch.zeros((), device=self.device)
        b = self._R_tilt @ _enu(az_m, alt_m)
        A = self._R_tilt @ torch.stack([torch.cos(az_m), -torch.sin(az_m), z])
        L = self._R_tilt @ torch.stack([-torch.sin(alt_m) * torch.sin(az_m),
                                        -torch.sin(alt_m) * torch.cos(az_m), torch.cos(alt_m)])
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
        """One vectorised SGP4 pass over [epoch, epoch+window] at dt_s -> (dt_s, n, alt, az_unwrapped)."""
        n = int(round(window_s / dt_s)) + 1
        secs = np.arange(n) * dt_s
        t0 = self._t0
        sec0 = t0.second + t0.microsecond * 1e-6
        times = self.ts.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute, sec0 + secs)
        alt, az, _ = (self.satellite - self._topos).at(times).altaz()
        az_unwrapped = np.unwrap(az.radians)        # continuous so linear interp can't jump at 0/2pi
        return dt_s, n, self._t(alt.radians), self._t(az_unwrapped)

    def _sat_altaz_at(self, seconds_list):
        """Interpolate the cached satellite ephemeris (radians) at the given sim-times."""
        dt_s, n, alt_tab, az_tab = self._sat_table
        idx = torch.tensor([s / dt_s for s in seconds_list], dtype=torch.float32).clamp(0, n - 1 - 1e-3)
        i0 = idx.floor().long()
        f = idx - i0
        return alt_tab[i0] * (1 - f) + alt_tab[i0 + 1] * f, az_tab[i0] * (1 - f) + az_tab[i0 + 1] * f

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

    def _psf(self, fb):
        sigma = self.cfg.psf_sigma_px
        rad = max(1, int(round(3 * sigma)))
        ax = torch.arange(-rad, rad + 1, dtype=torch.float32, device=fb.device)
        k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k1 /= k1.sum()
        kern = torch.outer(k1, k1)[None, None]
        return torch.nn.functional.conv2d(fb[None, None], kern, padding=rad)[0, 0]

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
        c_alt, c_az, c_mag = star_alt[keep], star_az[keep], self.star_mag[keep]

        fracs = [(i + 0.5) / substeps for i in range(substeps)]
        if self.satellite is not None:                       # interpolate the precomputed ephemeris
            sat_alt_all, sat_az_all = self._sat_altaz_at(
                [seconds_from_epoch + f * exposure_s for f in fracs])
            sat_mag = self._t([c.target_mag])

        fb = torch.zeros((c.height, c.width), dtype=torch.float32, device=self.device)
        for i, frac in enumerate(fracs):
            az = enc_az_rad + rate_az_rad_s * frac * exposure_s
            alt = enc_alt_rad + rate_alt_rad_s * frac * exposure_s
            b, A, L = self.boresight_basis(az, alt)
            if self.satellite is not None:                   # stars cached+culled; satellite batched
                s_alt = torch.cat([c_alt, sat_alt_all[i:i + 1]])
                s_az = torch.cat([c_az, sat_az_all[i:i + 1]])
                mag = torch.cat([c_mag, sat_mag])
            else:
                s_alt, s_az, mag = c_alt, c_az, c_mag
            px, py, vis = self.project(s_alt, s_az, b, A, L)
            flux_e = c.mag_flux_scale * (10.0 ** (-0.4 * mag)) * exposure_s / substeps
            self._splat(fb, px[vis], py[vis], flux_e[vis])

        fb = self._psf(fb)
        fb = torch.clamp(fb + c.sky_bg_rate_e * exposure_s, min=0.0)   # signal + sky bg (electrons)
        # Shot noise ~ Normal(lambda, sqrt(lambda)): a fast Gaussian approximation of Poisson
        # (exact at these electron counts, ~20x faster than torch.poisson, which dominated the
        # render), with read noise added in quadrature.
        fb = fb + torch.randn_like(fb) * torch.sqrt(fb + c.read_noise_e ** 2)
        adu = torch.clamp(torch.round(fb * c.adu_per_e), 0, 4095).to(torch.int32)
        val16 = (adu << 4)                                             # 12-bit -> 0xfff0 container
        return val16.cpu().numpy().astype(np.uint16)
