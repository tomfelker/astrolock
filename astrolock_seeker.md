# AstroLock Seeker

A simpler, closed-loop tracker for **any smoothly-moving bright object** (satellites,
the ISS, planes, rockets) using two cameras: a wide-field **guide camera** for finding
and tracking the target, and a **main camera** on the telescope for recording it.

Seeker deliberately ignores the hard problems that the main AstroLock app solves
(precise alignment, TLE propagation, a full world model). Instead it closes the loop
**in guide-camera pixel space**: detect the mover, drive it to the pixel that maps to
the center of the main scope, and record the main camera the whole time.

This document is the architecture/MVP plan. It is intentionally scoped down — see
[Out of scope (for now)](#out-of-scope-for-now) for the pile of fun ideas we are
deferring until the MVP stands up.

## Why this can be simple

The key insight: a pixel-space PID drives the target-to-boresight error to zero
**regardless of the camera's angle-per-pixel or any sky model**, as long as the sign of
the response is right and the loop gain is stable. We do not need alignment, RA/Dec, or
ephemerides for the control loop at all.

The only places real-world geometry sneaks back in:

- **Gimbal compensation.** On an AltAz mount the commanded azimuth rate must be divided
  by `cos(altitude)` to keep the *effective* loop gain constant as the target climbs
  toward zenith. That needs a rough altitude — hence the one calibration step that
  matters. (This is exactly what `kisstracker.py` already does.)
- **Feedforward while slewing.** To search the *expected* area for the target after
  we start moving, and to lead a fast mover instead of perpetually lagging it, we want
  the guide camera's angle-per-pixel. A guess from focal length + pixel pitch is fine;
  it can be refined later.

## Naming and process topology

A **pipeline of small, file-driven filters**, in the Unix tradition: each stage reads
files, writes files, and is independently runnable, crash-resilient, and testable offline
against old captures. Only the live-control path (GUI ⇄ backend) uses a socket.

| Process                 | Role                                                                 |
|-------------------------|---------------------------------------------------------------------|
| `astrolock_seeker_cam`     | One instance **per camera**. Captures continuously to a `.ser` file plus a per-frame metadata sidecar. Knows nothing about tracking. |
| `astrolock_seeker_detect`  | One instance **per camera of interest** (guide, at least). Reads a `.ser` (tailing live files, rolling to newer ones), finds bright/moving blobs, and writes a `detections` sidecar. Pure image→json; no mount, no sky model. |
| `astrolock_seeker_backend` | The brain. Launches & monitors the cams (and detectors), consumes the latest **detections**, runs target selection + the control loop, drives the mount over serial, and broadcasts state. |
| `astrolock_seeker_gui`     | Dear PyGui frontend. Reads `.ser` tails for live views and `detections`/`state` sidecars for overlays; sends live commands to the backend. Also a standalone **playback** viewer. |

The data flow is a chain: `cam` writes `.ser` + `.frames.jsonl`; `detect` reads those and
writes `.detections.jsonl`; `backend` reads detections, drives the mount, and writes
`.state.jsonl`; `gui` reads `.ser` + detections + state and sends commands to the backend.
The backend also launches/monitors the cam and detect processes (a supervision
responsibility, not a data path).

Every stage that consumes another's output is a **follower**: it tails a growing file and
rolls over to the next file when the current one ends (see below). That single behavior is
what makes the whole chain work identically live and in replay.

### Where the code lives (hybrid approach)

New package `astrolock/seeker/`, runnable as:

- `python -m astrolock.seeker.cam --config ...`
- `python -m astrolock.seeker.detect --config ... --ser ...`
- `python -m astrolock.seeker.backend --session ...`
- `python -m astrolock.seeker.gui --session ...`

(The module names map to the `astrolock_seeker_{cam,detect,backend,gui}` shorthand.) The
shared follower/SER/CV helpers are plain library modules in the same package, imported by
whichever processes need them.

Per the "hybrid" decision, we **import proven low-level bits rather than reimplement**,
but keep the control loop standalone (we do *not* route through `model/tracker.py` or its
modes):

- **Mount protocol** — reuse the NexStar AUX encoding from
  [`astrolock/model/telescope_connections/celestron_nexstar_hc.py`](astrolock/model/telescope_connections/celestron_nexstar_hc.py)
  (`_serial_send_axis_rate_cmd`, `_serial_read_axis_position_radians`, the careful
  `thinking_mid_time` timestamping). We may wrap a thin connection rather than use the
  full `TelescopeConnection`/`Tracker` machinery.
- **SER** — Seeker carries its own self-contained SER reader + writer in the package; no
  dependency on tensorez. The SER format is trivial (a 178-byte header + raw frames), so we
  copy/adapt the small amount of header/frame parsing logic rather than importing it. (The
  tensorez version is plain numpy and read-only anyway.)
- **Guide-cam CV** — lift helpers from [`focus.py`](focus.py): the `zwoasi` capture
  setup, `raw_to_bayer_planes`, `center_of_mass_offset`, `crop_image`,
  `indices_of_global_max`, and (eventually) the Sidewinder-style `generate_audio`
  lock-on tone.

The old Tkinter GUI under `astrolock/gui/` is considered **deprecated** for this effort;
Seeker's GUI is Dear PyGui and separate.

## SER as IPC + archive

`.ser` files are both the on-disk archive **and** the live transport between stages (cam →
detect, cam → gui). Rationale: dead simple, cross-platform, performant, and — crucially —
**crash-resilient**: if anything dies, the captured data is already on disk in a
standard, usable form.

Design rules:

- **Growing files with a sentinel frame count.** The cam writes a normal SER header but
  sets `frame_count = INT_MAX (0x7FFFFFFF)` while recording, and seeks back to patch the
  true count on clean shutdown. **All Seeker tools must tolerate `frame_count == INT_MAX`**
  and instead compute the real number of complete frames from the file size:
  `available = (file_size - header_size) // bytes_per_frame`.
- **The metadata sidecar is the commit point.** Each cam writes a companion
  `<ts>_<role>.frames.jsonl`, one line per frame, each self-identifying its `(ser, index)`
  and naming its successor (see [the spine](#the-sidecar-is-the-spine-followers-chain-along-it)).
  The cam appends the line **only after** the pixel bytes for that frame are flushed, so a
  follower never sees a torn frame. (This also keeps the `.ser` itself pure pixels and avoids
  the SER timestamp-trailer, which assumes a known frame count we don't have mid-capture.)
- **Followers read the tail.** A follower computes the latest committed frame from the
  sidecar and reads those pixel bytes from the `.ser` at the right offset — no locking on
  the live path. Prefer positioned `seek`+`read` for the growing tail (see Latency below).

**Cross-file ordering — guaranteed live, validated on recovery.** The commit point relies on
the `.ser` pixels being visible before the sidecar line that references them, across two
separate files. Your instinct is right: officially no, in practice yes — but it's worth being
precise about *which* guarantee. For the **live, same-machine** path it actually holds by
construction: a single-threaded producer flushes the pixels (a `write()` into the page cache)
before flushing the sidecar line, so any process that observes the sidecar line is causally
after the pixel write — same unified page cache, program order, no `fsync` needed. What
filesystems do **not** guarantee is cross-file *durability* ordering across a crash/power
loss: afterward the sidecar line might be on disk while the referenced pixels aren't (or
vice-versa). The fix is on the read side, not via `fsync`: trust a sidecar line only if the
`.ser` is physically long enough to contain that frame
(`file_size >= header + (index + 1) * bytes_per_frame`); otherwise wait (live) or stop at the
last consistent frame (recovery). That same check also handles a torn final frame from an
abrupt kill.

### Latency and throughput

The file-as-IPC path adds essentially no latency, because cross-process visibility goes
through the OS **page cache**, not the disk: bytes a producer writes are visible to a
follower's reads from RAM immediately. `fsync` is about *durability*, not visibility, and is
never on the hot path. Visible cross-process latency is sub-millisecond — dwarfed by the
~150 ms serial round trip and the exposure/processing budget. Three disciplines keep it that
way:

- **Flush userspace buffers at the commit point.** Bytes stuck in a producer's stdio /
  Python buffered writer aren't visible to other processes. The cam flushes the pixel bytes,
  then flushes the sidecar line — that ordering *is* the commit point, and it's cheap (a
  handoff to the kernel, no disk wait).
- **Prefer `seek`+`read` over `mmap` for the growing tail.** A memory map's length is fixed
  at creation, so following a growing file means re-mapping (awkward on Windows). Positioned
  reads are simpler and just as fast from cache; reserve `mmap` for random access into
  finalized files.
- **Wake cheaply — and just poll.** There is no portable way to *block* on a regular file
  growing: `read()` past EOF returns 0 immediately, and `select`/`poll`/`epoll` always report
  regular files as ready (`epoll` even rejects them with `EPERM`). Real options are (a) OS
  change-notifications — `inotify` / `kqueue` / `ReadDirectoryChangesW`, wrapped by Python's
  `watchdog` — which are a *wakeup hint only* (they coalesce and can drop, so you still loop
  "read until EOF" on wake); (b) a select()-able **doorbell** (pipe / `eventfd` / socket) the
  producer pokes after each commit, keeping the files as the durable archive; or (c) just
  **poll** at a small interval. At ~10 fps a 1–2 ms poll adds <5% of a frame and is the same
  trivial code on every OS — that's the MVP choice; notifications/doorbell are a later
  optimization only if a profile ever says so. Latency added is bounded by the poll cadence,
  not the OS.

The real constraint is **sustained write bandwidth**, not latency: a 3840×2160×16-bit frame
is ~16 MB (~160 MB/s per camera at 10 fps, more at higher rates), and if capture outruns
what the drive can flush, dirty pages pile up and the OS eventually throttles the writes
themselves. That's a storage-sizing question (NVMe, guide-cam ROI/binning) — independent of
the IPC style.

In practice only the **main** stream needs to be a full-rate keeper. The **guide** stream
still flows through files (so `detect` can read it), but it can be:

- **Low bandwidth** — the guide cam detects point sources, not pretty pictures, so bin/ROI
  it hard; its data rate is a fraction of the main cam's.
- **Rolling / ephemeral** — a janitor unlinks finalized guide segments older than a short
  window (the chained-segment design makes this trivial: just delete old `<ts>_guide.*`
  pairs). Crash-resilience is preserved for the recent window; you simply don't hoard frames
  you'll never look at.

Persisting the full guide stream becomes worthwhile only later, e.g. for plate-solving — at
which point it's a config flag (retention = keep) rather than an architecture change.

### The sidecar is the spine; followers chain along it

The `.frames.jsonl` sidecar — not the `.ser` — is the authoritative stream. Each record
identifies the frame it describes by `(ser, index)` and names its successor — but almost
none of that is actually written, because **everything resolves from implied defaults and a
field appears only to override its default:**

- **`ser`** defaults to the sidecar's own stem (`<ts>_<role>.frames.jsonl` ⇒
  `<ts>_<role>.ser`). Stated only to point at a *different* `.ser` (i.e. at a rollover).
- **`index`** defaults to the previous record's index + 1 (first record ⇒ 0). Stated only to
  resync/verify.
- **`next`** (the successor) defaults to `(same ser, index + 1)`. Stated only when it differs
  — which is exactly the rollover signal: a cam that splits at a size cap (or starts a fresh
  recording) writes one record whose `next` names the new `.ser`, and the follower rolls over.

So a frame reference is `{ "ser": ..., "index": ... }` with both fields optional; in the
common case a record carries neither and is pure metadata.

Every consuming stage is a *follower* of this spine, and they all need the same behavior, so
it's **one shared utility** (a library module in `astrolock/seeker/`, not a process):

1. **Tail the sidecar.** Read records as they're appended (filling in the implied `ser`/
   `index`), tolerating a trailing partial line by waiting for it to complete. The presence
   of a record is the commit point — its frame's pixels are already flushed to the resolved
   `.ser`, so a follower never sees a torn frame and never consults the `INT_MAX` SER header.
2. **Follow the successor.** At the latest record, resolve its successor. When the successor's
   `ser` differs from the current one, open that `.ser` (and its sidecar, by the same stem
   rule) and continue. A follower started cold seeks the newest existing chain by timestamp
   and joins it.

Because a "growing live file" and a "complete old file" are read by the *exact same code*,
running any stage on last week's capture is identical to running it live — which is the
whole point: `detect` and the GUI can be developed and tuned entirely offline.

### Config vs. captured data — two separate trees

Config and captured data have opposite lifecycles, so they live apart:

- **`config/`** — stable, hand-editable input. **Not timestamped.** This is what you tweak
  between sessions and check into version control if you like. The backend may *update*
  files here in place (e.g. writing back calibration), but they keep their plain names.
- **`sessions/<ts>/`** — disposable captured output, everything timestamp-prefixed for
  age-based purging.

```
config/
  guide.json            # one camera config per file; passed to a cam process by path
  main.json
  detect.json           # detector tuning (per-role overrides allowed)
  seeker.json           # mount, control, calibration, network (the backend's config)

sessions/20260624T210312Z/
  20260624T210312Z_guide.ser              # guide camera pixels (cam)
  20260624T210312Z_guide.frames.jsonl     # per-frame metadata + commit point (cam)
  20260624T210312Z_guide.detections.jsonl # bright/moving blobs per frame (detect)
  20260624T210312Z_main.ser               # main camera pixels (cam)
  20260624T210312Z_main.frames.jsonl      # per-frame metadata + commit point (cam)
  20260624T210312Z_state.jsonl            # backend telemetry stream (backend)
```

The sidecar naming is `<ts>_<role>.<kind>.jsonl` where `<kind>` is `frames` (cam) or
`detections` (detect) — so a role's pixels, frame metadata, and detections all share the
same `<ts>_<role>` stem and sort together.

**Every captured filename starts with a UTC timestamp** (ISO-8601 basic, `Z`-suffixed) so
files sort chronologically and purge with a simple glob/cutoff — no need to open or parse
anything. The session directory itself is timestamp-named (the session start), and each
file also carries its own start timestamp so it stays self-describing if moved out of the
directory. Optionally the backend can drop a timestamped *snapshot* of the effective config
into the session dir for reproducibility — that's a copy of the record, distinct from the
live `config/` inputs.

## Process independence (design principle)

Each process should be useful on its own; full orchestration is the convenient case, not a
requirement. Keeping the seams clean here is a stretch goal, but it shapes the interfaces:

- **`astrolock_seeker_cam`** takes a config file path as a command-line arg and runs
  **fully standalone** — point it at a camera config and it captures `.ser` +
  `.frames.jsonl` with no backend present. It is, in effect, a robust recorder.
- **`astrolock_seeker_detect`** takes a `.ser` (live or old) and a config, and emits a
  `detections.jsonl`. Run it on last week's capture to develop and tune detection with no
  cameras, no mount, no backend.
- **`astrolock_seeker_backend`** can run **headless** (no GUI): auto-create a session,
  launch/attach the cams + detectors, and capture (and optionally track, using the
  persisted calibration in `config/seeker.json`). All control is reachable over the command
  socket, so a GUI is optional.
- **`astrolock_seeker_gui`** has a **playback mode**: point it at an existing `sessions/<ts>/`
  and it replays the `.ser`/`.jsonl`/`state.jsonl` with **no backend and no sockets**. Live
  mode is the same viewer plus a connection to the backend's command socket. The viewer
  code path is identical for live tailing and historical playback (both just read growing-
  or-complete files).

This is why the file formats above are the contract, not the socket: anything essential is
on disk, and the socket only carries *live control*.

## Per-process detail

### `astrolock_seeker_cam`

Minimal, robust, fast. One process per camera, and a **standalone-useful recorder** —
point it at a config and it writes `.ser` segments; everything else is reconfigurable live.

- Input: source config (camera index/ROI/exposure/gain/bayer or sim) + capture settings.
- Loop: `capture()` → append raw bytes to the current `.ser` → append a `.frames.jsonl`
  line (incl. the `important` flag). Keep heavy processing out of here.
- **Settings** (CLI sets initial; the `--control-file` JSONL changes them live, read by a
  small reader thread — works for a file or `-`/stdin):
  - `frame-limit` (-1 unlimited): frames for the current file before rolling to the next.
  - `file-limit` (1; -1 unlimited): how many files to capture; exit when 0.
  - `important` (1; 0 = not recording): written per frame; the backend deletes only files
    with **no** important frame.
  - `exposure`/`gain`/`fps`: live-settable.
  - `{"stop": true}` (or `file-limit` 0) ends it.
- Files are timestamped per segment (`<segment_stamp>_<role>.ser`, ms resolution so they
  sort), so the cam can roll over without clobbering. No stop/pause flag-files — it stops
  via its control channel.

### `astrolock_seeker_detect`

A pure image→json filter. One per camera you care about (guide for sure; the main cam is
optional). No mount, no sky model, no sockets — just files in, files out.

- Input: a `.ser` to follow (and its `.frames.jsonl` for timestamps) + a detect config.
- Per frame, emit one line to `<ts>_<role>.detections.jsonl` listing candidate blobs:
  position (sub-pixel), brightness/score, size, and a `moving` flag.
- Two cheap signals, both local to this process since it has consecutive frames:
  - *bright blobs* — threshold + local-max / center-of-mass (reuse `focus.py` helpers).
  - *movers* — frame-subtraction vs the previous frame or a running median, to flag which
    blobs changed. Convenience for picking the target out of a field of stars; may not even
    be required.
- Deliberately **dumb about intent**: it does not know the boresight, the chosen target, or
  that the mount is slewing. It reports everything it sees. Temporal association and gating
  to the expected region are the backend's job (it has the mount motion + angle-per-pixel).
- Because it only reads files, it runs identically on a live capture or an old one — the
  primary place we develop/tune detection offline.
- *Later:* detection during a slew is much easier if it knows the **image-space slew rate**
  (so it can distinguish the real mover from star-field streaking, or shift-and-subtract).
  That hint could be fed in as another file stream the backend writes (preserving
  files-in/files-out purity), but it's deferred — the MVP detects while roughly stationary.

### `astrolock_seeker_backend`

The brain. Single control loop, no GUI dependencies, **no raw CV** (that's `detect`).

1. **Lifecycle:** launch the cam + detect subprocesses (or attach to already-running ones
   for resilience) and monitor their health.
2. **Select / track:** follow the guide `detections.jsonl`. Associate candidates frame to
   frame; once the user (or auto-pick) chooses one, gate to the **expected region**
   (predicted from last position + estimated velocity, shifted by known mount motion via the
   guide angle-per-pixel) and lock the best match. Output: target pixel + estimated pixel
   velocity.
3. **Control:** target-pixel error relative to the **boresight pixel** → angular error via
   guide angle-per-pixel → PID, plus **feedforward** from estimated target angular
   velocity → **÷ cos(altitude)** on azimuth → axis rates → serial to mount. Mirrors the
   existing `desired_rate + error*gain` shape in `model/pid.py`, extended past P-only.
4. **Broadcast:** append to `state.jsonl` (mode, chosen target pixel, error vector,
   commanded rates, altitude, loop timings, per-camera/detector health). Persisted +
   tailable.

**As built (closed loop in sim).** The backend talks to a **`Mount`** *driver* and treats sim
and real identically: `set_rates()`, `get_state()` (pose + rates), and `get_site()` (lat/lon/
elev + epoch — like the mount's GPS). `SimMount` is a real driver, not just an integrator: its
own loop runs at `--mount-update-hz` with speed + acceleration limits (periodic error etc. can
follow) and reports a configurable test site/clock; the backend feeds that site to the sky-sim
camera, so the simulated sky matches where/when the (simulated) mount thinks it is. Everything
runs in real time (sim time = epoch + elapsed wall-clock); if the synth lags it's just a lower
effective framerate (the cam renders the latest mount state). A global time-scale is deferred --
when added it must multiply *all* elapsed-monotonic-time reads uniformly. `CelestronMount` drives the
real NexStar mount on a dedicated serial thread (reusing `model/telescope_connections`; only
that thread touches the port — the Prolific drivers BSOD on multi-threaded access; real GPS read
is still TODO). `--mount` selects them. The controller
(`controller.PixelTracker`) closes the loop in guide pixels: nearest-blob association with a
gate + coast, then **PI** on the position error (the mount is itself an integrator, so PI →
zero steady-state lag on a constant-velocity target; the integral settles to the target's sky
rate and is clamped to the max motor rate). A **dead-zoned derivative brake** opposes only
image speed above a threshold, so it damps the acquisition slew / motion blur without
throttling a locked fast mover (whose target is held still in-frame). The detection→rate
mapping uses the cam's `bin` (frame sidecar) × optics `rad/px`. Azimuth is `÷cos(alt)` gimbal-
compensated (constant az loop gain, correct sign past 90°); within a small zenith zone az is
zeroed (chasing the singularity is futile) while altitude tips over and the loop re-acquires on
the far side — so it tracks straight through a near-zenith pass. **Not yet built:** boresight
calibration (hold the target at a calibrated boresight pixel rather than the frame centre).

### `astrolock_seeker_gui` (Dear PyGui)

- Live views of guide + main from the `.ser` tails (as a follower).
- Overlays, each from a file so they replay identically: candidate **boxes** from
  `detections.jsonl`, and boresight/locked-target/error from `state.jsonl`. Shape is a
  convention worth keeping: **square boxes = found in the image**; circles and lines =
  derived from geometry / the world model (a future source — see Out of scope).
- **Playback mode** needs none of the live processes: point it at a `sessions/<ts>/` and it
  draws everything from the three file kinds. Live mode is the same plus a backend socket.
- Commands to backend (live only): select candidate, **GO** (engage tracking), stop,
  calibration steps, exposure/gain. Sent over a **localhost socket**.

**The control loop must never block on the socket.** Whoever owns the loop (backend) reads
commands non-blocking (e.g. `select()`/`setblocking(False)`, or a reader thread draining
into a queue the loop polls). A slow or dead GUI can never stall tracking or mount I/O.
The same applies to the GUI side reading telemetry. For the simpler cam control surface, a
plain OS pipe / FIFO is a fine alternative to a socket — same non-blocking rule.

## Data formats

> **These examples are illustrative, not a spec to implement up front.** They show where the
> design is heading and assume features well beyond the MVP. When building, start with the
> few fields a stage actually needs and **add the rest only as needed** — the formats are
> additive (readers ignore unknown keys), so growing them later is free. Don't write a field
> nothing reads yet.

All structured data is JSON / JSON-lines. Angles are stated in their units per-field rather
than globally — pixel space in `px`, sky angles in `deg`, rates in `arcsec/s` (matching the
NexStar layer), times in two forms: `t_mono_ns` (a monotonic `perf_counter_ns`, for
latency/loop math) and `t_utc` (ISO-8601 `Z`, for wall-clock).

### `config/<role>.json` — one per camera (CLI arg to a cam process)

Everything a `astrolock_seeker_cam` needs to run standalone. `optics` lives here too so the
cam file fully describes one camera; the backend reads these for angle-per-pixel.

```jsonc
{
  "role": "guide",                       // "guide" | "main"; also the basename -> <ts>_guide.ser
  "driver": "zwoasi",
  "identity": { "name": "ZWO ASI678MC", "index": 0 },  // match by name, fall back to index
  "roi":   { "x": 0, "y": 0, "width": 3840, "height": 2160, "bin": 1 },
  "image_type": "RAW16",                 // RAW8 | RAW16 | RGB24
  "bayer": "RGGB",                       // or "MONO"
  "exposure_us": 2000,
  "gain": 190,
  // optics: angle_per_pixel may be given directly or derived from focal length + pitch
  "optics": { "focal_length_mm": 200, "pixel_pitch_um": 2.0, "angle_per_pixel_arcsec": null }
}
```

A cam run is then just: `python -m astrolock.seeker.cam --config config/guide.json
--out-dir sessions/<ts>/` (when launched by the backend it's handed the session dir; run
solo it can default to a fresh one).

### `config/seeker.json` — the backend's config

Mount, control, persisted calibration, and where to find the cameras. **Not timestamped**;
the backend may rewrite `calibration` in place after the calibration steps so a later
headless run can track without redoing them.

```jsonc
{
  "cameras": ["config/guide.json", "config/main.json"],  // paths the backend launches
  "mount": {
    "url": "celestron_nexstar_hc:COM4",   // reuses the model layer's URL scheme
    "max_rate_arcsec_s": [14400, 14400],  // [az, alt] safety clamp (~4 deg/s)
    "axis_sign": [1, 1]                    // flip if a positive command moves the wrong way
  },
  "calibration": {
    "altitude_offset_deg": null,           // from the "power on level" step; null until done
    "boresight_px": null,                  // [x, y] in GUIDE frame; null until done
    "calibrated_utc": null                 // when it was last set, for staleness judgement
  },
  "control": {
    "loop_hz_target": 10,
    "pid": { "p": [1.0, 1.0], "i": [0.0, 0.0], "d": [0.0, 0.0] },  // per [az, alt]
    "feedforward": true,
    "gimbal_compensation": true
  },
  "network": {
    "command_socket": { "host": "127.0.0.1", "port": 0 }  // 0 = OS-assigned; see discovery note
  }
}
```

When `port` is `0` the OS assigns one at runtime; the backend publishes the actual port in
its first `state.jsonl` line (and/or a small `<ts>_backend.json` in the session dir) so the
GUI discovers it without a fixed port collision — keeping with "the disk is the contract".

### `config/detect.json` — detector tuning (CLI arg to a detect process)

```jsonc
{
  "threshold": 0.15,            // fraction of full-scale to count as a blob
  "min_blob_px": 2,             // ignore single hot pixels
  "max_candidates": 16,         // cap per frame
  "subtract": "prev",           // "none" | "prev" | "median"; movers light up after subtraction
  "median_window": 30           // frames, when subtract == "median"
}
```

### `<ts>_<role>.frames.jsonl` — per-frame metadata sidecar / spine (cam → everyone)

One line appended per frame, **after** that frame's pixel bytes are flushed to its `.ser`
(the commit point). `ser`/`index`/`next` are all implied (see
[the spine](#the-sidecar-is-the-spine-followers-chain-along-it)); the common record is pure
metadata:

```jsonc
{ "t_mono_ns": 90123456789, "t_utc": "2026-06-24T21:05:48.213Z",
  "exposure_us": 2000, "gain": 190, "temp_c": 12.5, "dropped": 0 }

// ...and the one record at a rollover adds an explicit successor:
{ "t_mono_ns": ..., "next": { "ser": "20260624T211500Z_guide.ser", "index": 0 } }
```

### `<ts>_<role>.detections.jsonl` — candidate blobs (detect → backend, gui)

One line per processed frame, referencing its source frame by `(ser, index)` — implied the
same way (the detections sidecar's stem ⇒ the same `.ser`; index ⇒ prev + 1), so the common
record just carries blobs. Lists every blob found — no notion of "the target", just what's
visible.

```jsonc
{ "t_mono_ns": 90123456789,
  "blobs": [
    { "id": 3, "px": [1864.2, 1135.0], "score": 0.91, "size_px": 4.0, "moving": true },
    { "id": 5, "px": [402.0, 880.7],   "score": 0.40, "size_px": 2.0, "moving": false }
  ] }
```

`id` is stable across frames only as far as `detect`'s own cheap association manages it;
the backend does the authoritative temporal tracking and lock.

### `<ts>_state.jsonl` — backend telemetry (backend → GUI)

One line per control iteration (or capped to e.g. the loop rate). Broadcast + persisted;
the GUI tails it. Nulls where a stage hasn't produced data yet.

```jsonc
{
  "t_mono_ns": 90130000000,
  "t_utc": "2026-06-24T21:05:48.300Z",
  "mode": "locked",                         // idle | searching | locked | calibrating | estop
  "guide": { "ser": "20260624T210312Z_guide.ser", "index": 1421, "age_s": 0.087 },  // frame this decision used + its age
  "boresight_px": [1920, 1080],
  "target_id": 3,                               // which detections-blob is locked, or null
  "target_px": [1864, 1135], "target_vel_px_s": [42.0, -18.0],
  "error_px": [-56, 55], "error_deg": [-0.031, 0.030],
  "altitude_deg": 51.4,
  "commanded_rate_arcsec_s": [180.0, -64.0],    // [az, alt], post gimbal-comp
  "loop_dt_s": 0.101, "serial_rtt_s": 0.150,
  "cameras": { "guide": { "alive": true, "fps": 11.8, "last_frame_age_s": 0.09, "dropped": 0 },
               "main":  { "alive": true, "fps": 11.6, "last_frame_age_s": 0.10, "dropped": 0 } },
  "detectors": { "guide": { "alive": true, "last_index": 1421, "lag_frames": 1 } }
}
```

Candidate boxes come from `detections.jsonl` (the GUI reads that directly); `state.jsonl`
only reports which one is *locked* (`target_id` / `target_px`) and the resulting control.

*As built*, the backend publishes a leaner line: `mode` (`idle|slew|estop|track|lost`),
`enc_az_deg`/`enc_alt_deg`, `rate_az_deg_s`/`rate_alt_deg_s`, `recording`, `tracking`,
`track_role`, `target_px` (the locked target in frame image space, or null), `sources`,
`capturing`, and `cameras`. Boresight/error/`÷cos(alt)` fields arrive with calibration.

### Command messages (GUI → backend)

Newline-delimited JSON over the localhost socket, each with a `type` and an optional
client-chosen `id` echoed in the ack. The backend drains these non-blocking and replies
fire-and-forget; **state changes are confirmed via `state.jsonl`, not the ack** (the ack
just says "received/valid").

```jsonc
// implemented:
{ "type": "set_rate", "az": <deg/s>, "alt": <deg/s> }            // manual slew (press-and-hold)
{ "type": "stop" }                                              // zero rates
{ "type": "estop" }                                             // zero + latch idle
{ "type": "record", "on": true }                                // mark cam frames important (kept)
{ "type": "capture", "role": "guide", "on": false }             // stop/(re)start a camera
{ "type": "set_source", "role": "guide", "source": "sky" }      // swap sim <-> real live
{ "type": "track", "role": "guide", "px": [x, y] }              // lock nearest blob + engage PI loop
{ "type": "untrack" }                                           // disengage tracking, stop
// aspirational (calibration, not yet wired):
{ "type": "calibrate", "step": "altitude_zero" | "set_boresight", "px": [x, y] }
```

**Two-tier control.** The socket above is the *live hub* (latency-sensitive, GUI ⇄ backend).
The backend in turn drives each camera over a **per-cam control JSONL** (`control_<role>.jsonl`),
which the cam tails — a file, consistent with everything else cams touch (no sockets, crash-
resilient, replay-free since the backend truncates it per launch). The backend translates the
socket commands into control lines:

```jsonc
// backend -> cam, in control_<role>.jsonl (merge semantics; only keys present change):
{ "important": 1 }          // from record on/off
{ "frame_limit": 600 }      // segment length / roll trigger
{ "file_limit": 0 }         // graceful shutdown ("enough files")
{ "stop": true }            // immediate finalize + exit
{ "exposure_us": 1000, "gain": 200, "fps": 15 }
```

So *record* sets `important`, *capture off* sends `{stop:true}` (cam finalizes + exits) while
*capture on* relaunches the cam (+ its detector); on exit the backend deletes any `.ser`
segment whose sidecar has no important frame, and removes the session if nothing important
remains.

## Calibration (MVP)

Deliberately rough:

1. **Altitude reference.** Power on with the scope level at the horizon, so the altitude
   encoder reads ≈ 0. Gives us altitude → enables `cos(alt)` gimbal compensation. Does not
   need to be precise.
2. **Boresight.** Point at a bright star, center it in the **main** camera, and record its
   pixel in the **guide** camera. That guide pixel is the target everything is driven to.
3. **Angle-per-pixel.** Compute from config (focal length + sensor pixel pitch) for both
   cameras. Optional later refinement: slew a known amount, measure the pixel shift.

## MVP milestones

Ordered so the early ones need **no hardware** (just an old `.ser`), thanks to the
file-driven pipeline.

1. **SER reader/writer + follower lib.** Self-contained SER reader + writer (INT_MAX
   sentinel, frame-flush before sidecar line; adapt the small format logic), and the shared
   tail+rollover follower. Validate by re-reading a capture written live.
2. **`seeker_detect` offline.** Run `detect` on an existing `.ser` → `detections.jsonl`.
   Tune thresholds/subtraction against real old captures. No cameras, mount, or backend.
3. **`seeker_gui` playback.** Dear PyGui viewer that follows a `.ser` and draws boxes from
   `detections.jsonl` (and overlays from `state.jsonl` when present). Works on old sessions;
   this is the main detection-tuning UI.
4. **Live capture plumbing.** `seeker_cam` capturing real ASI cameras (two instances);
   `seeker_backend` launching cam+detect, broadcasting health, control loop stubbed. GUI
   now also runs live (same code as playback).
5. **Mount in the loop.** Backend talks serial to the mount; manual rate nudges from the
   GUI; gimbal compensation; verify sign conventions against the live view.
6. **Closed loop.** Backend selection/lock from detections + pixel-space PID + feedforward
   driving the target to boresight; GO/stop from the GUI; main cam recording throughout.
7. **Calibration UX.** The three calibration steps, driven from the GUI, persisted to
   `config/seeker.json`.

## Out of scope (for now)

Parked until the MVP works, but worth keeping in view:

- Shared-memory ring buffer for an even-lower-latency live path (SER stays the archive).
- Motion-model feedforward beyond constant angular velocity (orbit fitting, launch
  profiles) — see the main README's "Future Directions".
- **Auditory lock-on feedback** — a definite goal: it should sound just like a Sidewinder
  as the target nears boresight. The tone generator already exists in `focus.py`
  (`generate_audio`), driving pitch from a "luckiness" scalar; here it'd be driven by the
  boresight error.
- **World-model overlays on the guide preview.** Beyond image-found targets (square boxes),
  draw geometry-derived markers using the **circle/line** convention, sourced from the main
  AstroLock world model once integrated: circles around distant landmarks for daytime
  alignment, circles around ADS-B aircraft, circles + predicted-track lines for satellite
  TLEs, etc. — all composited over the guide camera view. (Each such source becomes another
  file the GUI overlays, same as `detections`/`state`.)
- Autoexposure / autogain on the cameras.
- Live exposure/gain retuning from the GUI.
- Sophisticated detection (motion-compensated subtraction during fast slews, multi-target
  disambiguation, star masking) — aided by an image-space slew-rate hint from the backend.
- Refined geometric calibration (solving angle-per-pixel and boresight from data).
- Integration back into / shared settings with the main AstroLock app.
- **Persistent guide capture + plate-solving**, giving absolute RA/dec per frame
  independent of mount alignment. Candidate solvers (skip the old local-astrometry.net
  Windows pain): **tetra3 / cedar-solve** (pure-Python, pip, in-process, builds its own
  FOV-specific DB — ideal for the *wide* guide cam and a control loop); **ASTAP** (fast
  single-binary CLI, robust, good with narrow FOVs, shell-out + parse `.wcs`); astrometry.net
  web via `astroquery` (no setup but network/slow, fallback only); Watney (local API-server
  option). Always feed a hint (mount RA/dec + pixel scale) for sub-second solves. Only the
  guide cam needs solving — the boresight offset maps it to the main scope's tiny FOV.
- **Plate-solve-assisted mount alignment.** A single solve gives one instantaneous
  (encoder ↔ known RA/dec) correspondence, *not* the mount model (encoder offsets + zenith
  tilt + optional cone errors), so several solves at different pointings are still needed —
  i.e. the existing 3-star alignment keeps its value. But plate solving feeds it *known*
  directions, removing the unknown-target identification that causes the main app's
  "more stars → worse solution" degeneracy. So plate solving augments the alignment
  optimizer rather than replacing it.
- **Orbit determination from a recorded pass.** A pass logs a dense, timestamped angle
  arc — enough for angles-only IOD (Gauss/Laplace) to recover all 6 elements in
  non-degenerate geometry (well-conditioned for high/curved passes; range/`a`/`e` weak on
  low horizon-skimmers; `B*` unobservable from one pass). Producing an actual TLE then means
  differential-correcting SGP4 mean elements against the observations. The main app's
  differentiable alignment optimizer is a natural fit for this (forward model: elements →
  SGP4 → observer + mount → predicted angles), and plate-solved guide frames are the
  cleanest input. Good for re-acquiring nearby passes; decays over days.
