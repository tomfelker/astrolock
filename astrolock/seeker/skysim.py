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
    mag_limit: float = 7.0
    mag_flux_scale: float = 1.0e6             # electrons/s for a mag-0 source
    sky_bg_rate_e: float = 80.0              # sky background electrons / pixel / s
    read_noise_e: float = 2.0
    adu_per_e: float = 0.05                   # gain: electrons -> 12-bit ADU

    # optional satellite target: (tle_line1, tle_line2, name), plus a magnitude
    target_tle: tuple = None
    target_mag: float = 1.0


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

    def _t(self, v):
        return torch.as_tensor(v, dtype=torch.float32, device=self.device)

    # --- geometry -----------------------------------------------------------

    def boresight_basis(self, enc_az_rad, enc_alt_rad):
        """True boresight unit vector b and tangent axes (A=az dir, L=alt dir), all (3,)."""
        c = self.cfg
        az_m = self._t(enc_az_rad - math.radians(c.az_offset_deg))
        alt_m = self._t(enc_alt_rad - math.radians(c.alt_offset_deg))
        b = self._R_tilt @ _enu(az_m, alt_m)
        b_alt = torch.asin(torch.clamp(b[2], -1.0, 1.0))
        b_az = torch.atan2(b[0], b[1])
        z = torch.zeros((), device=self.device)
        A = torch.stack([torch.cos(b_az), -torch.sin(b_az), z])
        L = torch.stack([-torch.sin(b_alt) * torch.sin(b_az),
                         -torch.sin(b_alt) * torch.cos(b_az), torch.cos(b_alt)])
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
        fb = torch.zeros((c.height, c.width), dtype=torch.float32, device=self.device)
        for i in range(substeps):
            frac = (i + 0.5) / substeps
            t_s = seconds_from_epoch + frac * exposure_s
            az = enc_az_rad + rate_az_rad_s * frac * exposure_s
            alt = enc_alt_rad + rate_alt_rad_s * frac * exposure_s
            b, A, L = self.boresight_basis(az, alt)
            s_alt, s_az, mag = self.sources_altaz(self._sf_time(t_s))
            px, py, vis = self.project(s_alt, s_az, b, A, L)
            flux_e = c.mag_flux_scale * (10.0 ** (-0.4 * mag)) * exposure_s / substeps
            self._splat(fb, px[vis], py[vis], flux_e[vis])

        fb = self._psf(fb)
        fb = torch.clamp(fb + c.sky_bg_rate_e * exposure_s, min=0.0)   # sky background
        fb = torch.poisson(fb) + torch.randn_like(fb) * c.read_noise_e  # shot + read noise
        adu = torch.clamp(torch.round(fb * c.adu_per_e), 0, 4095).to(torch.int32)
        val16 = (adu << 4)                                             # 12-bit -> 0xfff0 container
        return val16.cpu().numpy().astype(np.uint16)
