"""
Camera-side sky renderer for Seeker (the optics/pixels half; propagation lives in sky_sim.py).

Given a (hidden, truth) tripod orientation + optics, an *encoder* pose, and the set of point-source
directions in the sky *at each exposure substep* (from the shared ephemeris), this renders a frame:
project sources to pixels (gnomonic / pinhole), sum across the substeps to get motion streaks (mount
slew), convolve a PSF, add shot+read noise, quantize to the 12-bit-in-16-bit container the real cam
produces.

Sources come in as topocentric ENU unit vectors (x=east, y=north, z=up) -- one per source per
substep -- so a star is static across the substeps while a fast satellite point moves; the exposure
streak is the boresight sweeping. Every source is a uniform point; the satellite's shape is just its
body points arriving as many sources. All math is torch (differentiable, for solving calibration).

The "truth" here (tripod tilt, encoder offsets, exact optics) is what the backend does NOT know --
that's what makes a closed-loop test meaningful.
"""

import dataclasses
import math

import numpy as np
import torch


@dataclasses.dataclass
class SkySimConfig:
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

    # rendering / noise
    psf_sigma_px: float = 1.3
    mag_flux_scale: float = 4.0e6             # electrons/s for a mag-0 source
    sky_bg_rate_e: float = 80.0              # sky background electrons / pixel / s
    read_noise_e: float = 2.0
    adu_per_e: float = 0.05                   # gain: electrons -> 12-bit ADU


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def _enu(az, alt):
    """Unit vector(s) in East-North-Up from azimuth (from north, toward east) and altitude."""
    ca = torch.cos(alt)
    return torch.stack([ca * torch.sin(az), ca * torch.cos(az), torch.sin(alt)], dim=-1)


def _next_fast_len(n):
    """Smallest m >= n whose only prime factors are 2,3,5,7 -- a size the FFT is fast at."""
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def ensure_cache(cache_dir='data/skyfield_cache'):
    """Download the skyfield ephemeris + Hipparcos catalog into the cache if missing. Call once
    before launching the sky_sim process. A no-op once present. (Only sky_sim needs the catalog now.)"""
    import os
    from skyfield.api import Loader
    from skyfield.data import hipparcos
    os.makedirs(cache_dir, exist_ok=True)
    loader = Loader(cache_dir)
    loader.timescale()
    loader('de421.bsp')
    with loader.open(hipparcos.URL):
        pass


class SkySim:
    def __init__(self, config=None, device='cpu'):
        self.cfg = config or SkySimConfig()
        self.device = device
        c = self.cfg
        self.f_px = c.focal_length_mm / (c.pixel_pitch_um * 1e-3)   # pixels per radian-ish
        self.cx, self.cy = (c.boresight_px if c.boresight_px
                            else (c.width / 2.0, c.height / 2.0))
        # hidden tripod rotation: mount-frame ENU -> true ENU
        self._R_tilt = (_rot_x(math.radians(c.zenith_pitch_deg))
                        @ _rot_y(math.radians(c.zenith_roll_deg))).to(device)
        self._otf = None                           # PSF transfer function, built lazily
        # FoV culling radius: half-diagonal of the frame + a margin for slew during the exposure
        half_x = math.atan(c.width * c.pixel_pitch_um * 1e-3 / (2 * c.focal_length_mm))
        half_y = math.atan(c.height * c.pixel_pitch_um * 1e-3 / (2 * c.focal_length_mm))
        self._cull_cos = math.cos(math.hypot(half_x, half_y) + math.radians(3.0))

    def _t(self, v):
        return torch.as_tensor(v, dtype=torch.float32, device=self.device)

    # --- geometry -----------------------------------------------------------

    def boresight_basis(self, enc_az_rad, enc_alt_rad):
        """
        True boresight unit vector b and camera tangent axes A (image right = the fixed alt rotation
        axis) and L (image up = d boresight / d alt), from the mount-frame orientation rotated by the
        tripod tilt, so they stay continuous past the zenith. Scalar enc_* -> (3,); an (S,) batch of
        poses -> (S, 3), so one call builds every substep's basis.
        """
        c = self.cfg
        az_m = self._t(enc_az_rad - math.radians(c.az_offset_deg))
        alt_m = self._t(enc_alt_rad - math.radians(c.alt_offset_deg))
        z = torch.zeros_like(az_m)
        rt = self._R_tilt.T
        b = _enu(az_m, alt_m) @ rt
        A = torch.stack([torch.cos(az_m), -torch.sin(az_m), z], dim=-1) @ rt
        L = torch.stack([-torch.sin(alt_m) * torch.sin(az_m),
                         -torch.sin(alt_m) * torch.cos(az_m), torch.cos(alt_m)], dim=-1) @ rt
        return b, A, L

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
        sigma = self.cfg.psf_sigma_px
        rad = max(1, int(round(3 * sigma)))
        ax = torch.arange(-rad, rad + 1, dtype=torch.float32, device=self.device)
        k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k1 /= k1.sum()
        return torch.outer(k1, k1), rad

    def _build_otf(self, h, w):
        kern, rad = self._psf_kernel()
        sz = (_next_fast_len(h + 2 * rad), _next_fast_len(w + 2 * rad))
        self._otf = torch.fft.rfft2(kern, s=sz)
        self._otf_sz, self._otf_rad, self._otf_hw = sz, rad, (h, w)

    def _psf(self, fb):
        h, w = fb.shape
        if self._otf is None or self._otf_hw != (h, w):
            self._build_otf(h, w)
        out = torch.fft.irfft2(torch.fft.rfft2(fb, s=self._otf_sz) * self._otf, s=self._otf_sz)
        r = self._otf_rad
        return out[r:r + h, r:r + w]

    def render(self, enc_az_rad, enc_alt_rad, rate_az_rad_s, rate_alt_rad_s,
               source_dirs, source_mag, exposure_s=0.1, substeps=6):
        """Render one frame -> uint16 (H, W) ndarray (12-bit data left-shifted into 16 bits).

        source_dirs: (T, S, 3) ENU unit dirs, one per source per substep (S == substeps).
        source_mag:  (T,) magnitudes. The boresight sweeps enc..enc+rate*exposure across the substeps.
        """
        c = self.cfg
        S = substeps
        s = torch.arange(S, dtype=torch.float32, device=self.device)
        fr = (s + 0.5) / S                                    # (S,) substep mid-fractions
        az_s = self._t(enc_az_rad) + self._t(rate_az_rad_s) * fr * exposure_s
        alt_s = self._t(enc_alt_rad) + self._t(rate_alt_rad_s) * fr * exposure_s
        b, A, L = self.boresight_basis(az_s, alt_s)           # each (S, 3)

        fb = torch.zeros((c.height, c.width), dtype=torch.float32, device=self.device)
        if source_dirs is not None and source_dirs.numel() > 0:
            dirs = source_dirs.to(self.device)                # (T, S, 3)
            mag = source_mag.to(self.device)                  # (T,)
            mid = S // 2
            keep = (dirs[:, mid, :] @ b[mid]) > self._cull_cos    # cull to the FoV (thousands -> dozens)
            dirs, mag = dirs[keep], mag[keep]
            if dirs.numel() > 0:
                denom = torch.einsum('tsj,sj->ts', dirs, b)   # (T, S) each source . its substep basis
                X = torch.einsum('tsj,sj->ts', dirs, A) / denom
                Y = torch.einsum('tsj,sj->ts', dirs, L) / denom
                phi = math.radians(c.roll_deg)
                cphi, sphi = math.cos(phi), math.sin(phi)
                px = self.cx + self.f_px * (X * cphi + Y * sphi)      # (T, S)
                py = self.cy - self.f_px * (-X * sphi + Y * cphi)
                vis = denom > 0
                flux = c.mag_flux_scale * (10.0 ** (-0.4 * mag)) * exposure_s / S   # (T,)
                flux = flux[:, None].expand(-1, S)
                self._splat(fb, px[vis], py[vis], flux[vis])

        fb = self._psf(fb)
        fb = torch.clamp(fb + c.sky_bg_rate_e * exposure_s, min=0.0)   # signal + sky bg (electrons)
        # Shot noise ~ Normal(lambda, sqrt(lambda)) (a fast, at these counts exact, Poisson approx),
        # plus read noise in quadrature.
        fb = fb + torch.randn_like(fb) * torch.sqrt(fb + c.read_noise_e ** 2)
        adu = torch.clamp(torch.round(fb * c.adu_per_e), 0, 4095).to(torch.int32)
        return (adu << 4).cpu().numpy().astype(np.uint16)     # 12-bit -> 0xfff0 container
