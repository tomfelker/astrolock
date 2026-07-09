"""
Camera-side sky renderer for Seeker (the optics/pixels half; propagation lives in sky_sim.py).

Given a (hidden, truth) tripod orientation + optics, an *encoder* pose, and the set of point-source
directions in the sky *at each exposure substep* (from the shared almanac), this renders a frame:
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
    psf_sigma_px: float = None                # residual blur (seeing/defocus/aberration) atop the Airy disc,
                                              # px. None = auto: 0 when the aperture is known (trust the
                                              # physical diffraction PSF), else 1.3 (the Gaussian IS the PSF).
    aperture_mm: float = 0.0                  # objective aperture; >0 -> physically-sized Airy-disc PSF
    psf_wavelength_nm: float = 550.0          # wavelength for the Airy disc (green mid-band)
    seeing_r0_m: float = 0.0                  # atmospheric Fried parameter r0 (m); >0 adds a seeing blur
                                              # (FWHM ~ 0.98*lambda/r0). Typical: 0.05 poor .. 0.2 excellent
    mag_flux_scale: float = 4.0e6             # electrons/s for a mag-0 source
    sky_bg_rate_e: float = 80.0              # sky background electrons / pixel / s
    read_noise_e: float = 2.0
    adu_per_e: float = 0.05                   # gain: electrons -> 12-bit ADU
    noise_refresh: int = 0                    # frames between redraws of the reused unit-noise buffer
                                              # (0 = never; the buffer is cropped at a random offset each
                                              # frame, which is ~free vs drawing randn per pixel)


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


def _render_tail(fb, noise, bg_e, read_var, adu_per_e):
    """The render's elementwise tail: add the sky background, apply signal-dependent shot noise + read
    noise (unit-normal field * per-pixel std sqrt(signal + read^2) -- the Poisson->Gaussian approx), and
    quantize to 12-bit ADU. One fusable elementwise chain; eager torch runs it as ~a dozen separate
    full-array passes, so torch.compile fusing it into a single kernel is a big win (see _apply_tail)."""
    fb = torch.clamp(fb + bg_e, min=0.0)
    fb = fb + noise * torch.sqrt(fb + read_var)
    return torch.clamp(torch.round(fb * adu_per_e), 0, 4095).to(torch.int32)


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
        self._noise_buf = None                     # reused unit-normal buffer (cropped per frame)
        self._noise_ctr = 0
        self._noise_margin = 256                   # random crop-offset range into the noise buffer
        self._tail = None                          # compiled render tail, built lazily on the first frame
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

    def _centred_offsets(self, sz):
        """(yy, xx) pixel offsets from the origin over the whole `sz` grid, in FFT (wrapping) order --
        0,1,..,n/2-1,-n/2,..,-1 -- so a kernel built on them peaks at index (0,0) and the convolution
        adds no shift."""
        H, W = sz
        fy = torch.fft.fftfreq(H, device=self.device) * H
        fx = torch.fft.fftfreq(W, device=self.device) * W
        return torch.meshgrid(fy, fx, indexing='ij')

    def _airy_otf(self, sz, r_null_px):
        """rfft2 of the Airy diffraction pattern I(r) = [2 J1(v)/v]^2 (first null at r_null_px) computed
        over the WHOLE (centred, wrapping) frame -- so there's no small square kernel to leave a faint
        square halo around bright stars; the pattern just decays to ~0 by the frame edge."""
        yy, xx = self._centred_offsets(sz)
        v = (3.831705970 / r_null_px) * torch.sqrt(xx * xx + yy * yy)
        inten = torch.where(v < 1e-9, torch.ones_like(v), (2.0 * torch.special.bessel_j1(v) / v) ** 2)
        return torch.fft.rfft2((inten / inten.sum()))

    def _gauss_otf(self, sz, sigma):
        """rfft2 of a full-frame centred Gaussian of stddev `sigma` px."""
        yy, xx = self._centred_offsets(sz)
        g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        return torch.fft.rfft2((g / g.sum()))

    def _seeing_sigma_px(self):
        """Atmospheric-seeing Gaussian sigma (px) from the Fried parameter r0: FWHM = 0.98*lambda/r0
        (rad) -> px via the plate scale (f_px = px/rad), /2.3548 to sigma. 0 when r0 is unset."""
        c = self.cfg
        if c.seeing_r0_m <= 0:
            return 0.0
        fwhm_px = 0.98 * (c.psf_wavelength_nm * 1e-9) / c.seeing_r0_m * self.f_px
        return fwhm_px / 2.354820045

    def _build_otf(self, h, w):
        # Build the PSF transfer function once (per frame size). We DON'T pad by a kernel radius: the
        # kernels are built full-frame and centred at the origin, so there's no shift and no square kernel
        # boundary, and a near-edge star's PSF just wraps to the opposite edge (cheap, fine for a star
        # field). sz only rounds up to the next 2,3,5,7-smooth length (2160 stays 2160, not a slow 2205).
        c = self.cfg
        sz = (_next_fast_len(h), _next_fast_len(w))
        # Residual Gaussian blur = aberration/defocus (psf_sigma_px, auto per policy) + atmospheric
        # seeing (from r0), added in quadrature.
        sigma = c.psf_sigma_px
        if sigma is None:                              # auto: pure diffraction when optics are known
            sigma = 0.0 if c.aperture_mm > 0 else 1.3
        sigma = math.hypot(sigma, self._seeing_sigma_px())
        if c.aperture_mm > 0:                          # physically-sized Airy disc, optionally blurred
            fnum = c.focal_length_mm / c.aperture_mm
            r_null_px = 1.2196699 * (c.psf_wavelength_nm * 1e-9) * fnum / (c.pixel_pitch_um * 1e-6)
            otf = self._airy_otf(sz, r_null_px)
            if sigma > 0:
                otf = otf * self._gauss_otf(sz, sigma)  # convolution = product in the frequency domain
        else:                                          # unknown optics: the Gaussian IS the PSF
            otf = self._gauss_otf(sz, max(sigma, 1e-3))
        self._otf = otf
        self._otf_sz, self._otf_hw = sz, (h, w)

    def _psf(self, fb):
        h, w = fb.shape
        if self._otf is None or self._otf_hw != (h, w):
            self._build_otf(h, w)
        out = torch.fft.irfft2(torch.fft.rfft2(fb, s=self._otf_sz) * self._otf, s=self._otf_sz)
        return out[:h, :w]

    def _apply_tail(self, *args):
        """Run the elementwise render tail, fused into one kernel by torch.compile (built lazily on the
        first frame). We gate the build on is_dynamo_supported() -- older torch RAISES 'Windows not yet
        supported' inside torch.compile() itself, which suppress_errors (a runtime-only setting) can't
        catch -- so it degrades to eager seamlessly, no flag. (Force plain eager with TORCHDYNAMO_DISABLE=1.)"""
        if self._tail is None:
            self._tail = _render_tail                      # eager fallback (default if compile is unavailable)
            try:
                import logging
                import torch._dynamo
                if torch._dynamo.is_dynamo_supported():    # False on older torch (Windows) -> stay eager
                    torch._dynamo.config.suppress_errors = True    # runtime graph breaks -> eager, quietly...
                    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)   # ...(no cl.exe / no Triton)
                    self._tail = torch.compile(_render_tail)
            except Exception:                              # torch too old to even have is_dynamo_supported
                self._tail = _render_tail
        return self._tail(*args)

    def _unit_noise(self, h, w):
        """A unit-normal (h, w) field via a random-offset crop of a precomputed buffer -- exact Gaussian
        statistics but ~free, vs drawing randn every pixel every frame (which dominated the render). The
        realizations are reused (fine for a sim -- per-pixel mean/variance stay correct, and random
        offsets keep frames decorrelated); set cfg.noise_refresh > 0 to periodically redraw the pool."""
        S = self._noise_margin
        if self._noise_buf is None or self._noise_buf.shape != (h + S, w + S):
            self._noise_buf = torch.randn(h + S, w + S, device=self.device)
            self._noise_ctr = 0
        elif self.cfg.noise_refresh and self._noise_ctr >= self.cfg.noise_refresh:
            self._noise_buf.normal_()
            self._noise_ctr = 0
        self._noise_ctr += 1
        dy = int(torch.randint(0, S + 1, (1,)))
        dx = int(torch.randint(0, S + 1, (1,)))
        return self._noise_buf[dy:dy + h, dx:dx + w]

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
        # Sky background + signal-dependent shot noise + read noise + quantize -- one big elementwise
        # chain (see _render_tail), fused into a single kernel by torch.compile when enabled. Scalars go
        # in as 0-d tensors so changing exposure/gain doesn't force a recompile. The unit-normal field is
        # a cheap reused-buffer crop, computed here (outside the compiled region).
        dev = fb.device
        noise = self._unit_noise(fb.shape[0], fb.shape[1])
        adu = self._apply_tail(fb, noise,
                               torch.tensor(c.sky_bg_rate_e * exposure_s, dtype=torch.float32, device=dev),
                               torch.tensor(c.read_noise_e ** 2, dtype=torch.float32, device=dev),
                               torch.tensor(c.adu_per_e, dtype=torch.float32, device=dev))
        return (adu << 4).cpu().numpy().astype(np.uint16)     # 12-bit -> 0xfff0 container
