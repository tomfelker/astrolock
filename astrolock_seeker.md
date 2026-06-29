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

## Running it

Everything runs from the repo root with the project venv. The simplest invocation needs no
hardware — it launches the whole pipeline (a simulated mount + sky camera, a detector, and the
GUI) on a baked-in **ISS test pass**:

```
python -m astrolock.seeker.backend
```

The GUI opens on the rising ISS, pointed roughly (not exactly) at it so acquisition is
exercised. **Left-click the bright ISS** to lock and track; it holds it through the near-zenith
crossing (altitude tips over, azimuth dead-zones at the singularity) and **right-click** stops.
The backend launches and monitors the cam + detector subprocesses and runs the ~20 Hz loop.

Useful flags (all on `astrolock.seeker.backend`):

- `--source synthetic` — a no-dependency moving-blob scene (no Skyfield/torch needed);
  `--source zwo` — a real ASI camera; default `sky` is the simulator.
- `--mount sim` (default) vs `--mount celestron --mount-url celestron_nexstar_hc:COM3` — the
  real NexStar mount (driver written, untested at the scope).
- `--epoch`, `--sky-tle-file`, `--sky-target-mag`, `--start-az-deg/--start-alt-deg` — pick a
  different pass/target/pointing; `--no-gui --duration N` for headless runs.
- `--max-rate-deg-s`, `--mount-accel-deg-s2`, `--track-ki`, `--track-damping`, `--track-kd`,
  `--track-max-px-s`, `--track-zenith-zone-deg`, … — mount + controller tuning.

Components are runnable standalone too (e.g. `python -m astrolock.seeker.cam --source sky`,
`python -m astrolock.seeker.gui --session sessions/<ts>` to replay a recorded session). Tests:
`python -m astrolock.seeker.tests.<name>` (ser, bayer, cam_follower, detect, gui_prep,
cam_control, controller, skysim).

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
   *(Preferred per the backlog: instead of a separate step, nudge the boresight offset live while
   tracking until the target centers in the main cam — that becomes the calibration, and absorbs
   mirror flop.)*
3. **Angle-per-pixel.** Compute from config (focal length + sensor pixel pitch) for both
   cameras. Optional later refinement: slew a known amount, measure the pixel shift.

## MVP milestones

Ordered so the early ones need **no hardware** (just an old `.ser`), thanks to the
file-driven pipeline.

> **Status (as built).** Milestones 1–6 are done and validated *in simulation*: the SER/follower
> libs, offline + live `detect`, the GUI (playback and live), live capture plumbing, the mount in
> the loop, and the closed loop (click-to-track PI + dead-zoned-D brake, azimuth `÷cos(alt)` gimbal
> compensation, zenith dead-zone, frame-timestamp-clocked control). Beyond the original plan we
> also built a physically-based **sky simulator** + a **simulated mount driver** (so the whole
> loop runs with no hardware), baked in an **ISS test pass**, and made the sim hit ~30 fps.
> **Remaining:** run milestone 5's serial path on the **real mount** (driver written, untested),
> and milestone 7 (**calibration UX** — boresight is currently just the frame centre).

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

## Backlog (brainstorm)

Captured ideas, not yet scheduled. Grouped by area.

### GUI
- **Live camera settings** per camera (guide + main): exposure time, frame rate, gain, bit
  depth. Exposure/gain/fps already go over the per-cam control channel; **bit depth / ROI need
  a cam relaunch** (format change). Each setting gets **+/- buttons to bump by a half f-stop**
  (×/÷ √2 on exposure / linear gain) for fast dialing-in.
- **Histograms** per camera, as a **readout** (judge exposure, spot clipping) — *not* a stretch
  control. The preview stays **WYSIWYG (no display stretch)** so exposure can be set accurately;
  you expose/gain the camera until it looks right rather than stretching pixels.
- **ROI (region of interest)** per camera (most useful on the narrow main cam): a checkbox + size
  text boxes to read out a sub-window (higher framerate, less data). Plus an **"auto ROI"** mode:
  once the target is tracked and near center, switch to a small ROI and **move it each frame to
  keep the target centered** — *electronic* following on the sensor while the mount tracks
  *mechanically*. Feasibility: ZWO can move the ROI start position live, and **moving (not
  resizing) keeps the `.ser` dimensions fixed**, so only the ROI origin changes — recorded
  per-frame in the frames sidecar (which already carries `roi`); a resize is a new segment. The
  backend tracks in **full-sensor coords** (`roi_origin + in-ROI pos × bin`) against the sensor
  boresight, so it still has the target's position even though it ideally stays centered in the
  window — i.e. two loops: the mount (coarse) and the ROI (fine).
- **Adjustable preview size**: the preview is currently the debayered half-res scaled to
  `--display-width` (default 640) — ~6× down from a 3840 sensor; make it a GUI control / larger
  default.
- **WYSIWYG display, no white balance.** Neutral WB on the camera (pristine raw — already the
  default); the display does *only* a standard linear→sRGB conversion (standard gamma), with **no
  per-channel display gains** (drop the current `wb_r`/`wb_b`). Casts are real (e.g. a baby-blue
  wall; foliage blown white by an IR spotlight with no IR-cut filter), and we want to see truth.
- **Playback source** (built — `cam --source playback --playback-ser <file>`): replays any
  recorded `.ser` through the *live* pipeline at its recorded cadence (`--playback-speed`,
  `--playback-loop`), so `detect`/GUI/tracking all run on it — the easy way to review a capture
  with detections overlaid (the GUI alone only shows the newest frame of a finalized file). TODO:
  a **GUI file picker** to choose the `.ser` to replay.
- **Zoomed target inset(s)**: a magnified live view of the tracked target (and/or the boresight),
  great for focusing and for eyeballing detection. **(stretch) integrate the focus/collimation
  tool** (`focus.py`): show its star-profile EMA + screw-turn guidance on the tracked target.
- **Reticles**: crosshair/circle at the boresight pixel; outline the **main camera's FOV** on
  the guide view (both follow from boresight + plate scale).
- **Alt/az grid** (and later RA/dec) drawn over the guide view from the mount pose + plate
  scale — effectively the first world-model overlay.
- **Auto-record when tracking starts** — a **per-camera** toggle; each camera independently
  records from lock to unlock.
- Two-camera layout throughout (guide + main panes), since both are coming.
- **Camera enumeration**: ask each driver which cameras it sees (zwo already has
  `list_cameras()`; generalize across drivers) and let the GUI assign physical cameras to roles
  (guide / main). Needed once we run two cameras.
- **Gamepad slewing**: reuse the main app's PS4/PS5 gamepad support for manual slew — it feeds the
  same `set_rate` command path the on-screen pad already uses.
- **2D slew pad** (replace the four direction buttons): a box whose circle always **displays the
  current commanded slew rate** (center = zero). The command is a **priority mux** — mouse
  click/drag in the pad > gamepad stick > tracking-loop output > zero. "Spring return" just means a
  source deactivates when released, so the displayed/commanded rate **falls through to the next
  active source**: e.g. mouse-nudge while tracking, release, and the **tracker resumes** (a
  momentary override, not a cancel — unlike today's "manual stops tracking"). Optional **log-scale
  axes** (fine near center, full slew at the edges). Implies a small **rate-source arbiter** in the
  backend replacing the current "`set_rate` cancels tracking" behavior.

### Sim camera
- **Optics + sensor presets**: rather than maintain our own library, **vendor a snapshot of
  Stellarium's Oculars data** (`ocular.ini`: `[ccd]` chip size/resolution/pixel size,
  `[telescope]` focal length/aperture — read with `configparser`), deriving pixel pitch + FOV.
  The contents are bare facts (no copyright concern); refresh the snapshot occasionally. Doubles
  as real-camera config later.
- **(stretch) Physically accurate star brightness**: electrons from aperture × QE × bandpass ×
  exposure × a zero-point, so sim ADU matches what a real sensor would record.
- **Render the ISS as a crude multi-point sketch** (~tens of points: truss/backbone + solar
  arrays + modules) rather than a single point, to convey its angular *size* in frame. Offset each
  body point from the satellite center by `body_meters / slant_range` in direction space and
  project through the normal pipeline, so the size is automatically distance-correct; spread the
  total flux over the points. Not meant to be accurate — a rough heuristic orientation (long axis
  ⟂ ground-track velocity) is enough. Mostly visible in a long-focal main-scope view; in the wide
  guide cam the ISS is genuinely ~1px.
- **(very stretch) Dynamic seeing via fine-step drizzle**: integrate each exposure at a fixed
  ~1 ms step, **lerping every source along its PWL world-path** (cheap — no per-step astrometry)
  and perturbing by a **time-correlated atmospheric tip/tilt** (common across the field for the
  small FoV; correlation time + amplitude as Fried-ish knobs). Integrating over the exposure then
  *produces* the seeing itself — a dancing/displaced star on short exposures (lucky-imaging),
  blurred into a seeing disk on long ones — replacing/augmenting the fixed Gaussian PSF. Bonus: it
  gives the tracker realistic centroid jitter to be robust against.

### Detection
- **Only fixed-source detection really matters.** The point is to *track*, which holds the target
  ~still in frame, so detecting a crisp **point source** is the case that counts. Streaked/moving
  detection only matters in the brief pre-lock acquisition transient (covered by the mover-finding
  below) — so tune for fixed points and don't over-engineer streak handling.
- **Works poorly on real, non-starfield scenes.** Indoors with a laser pointer: finds nothing
  (not even the dot) or false-positives on a flat textured surface. On a real night capture
  (3840×2160 RGGB, 8 mm lens, out a shed door, auto-exposure 0.2 s / gain 400 — many stars incl.
  the Big Dipper, plus two very dim moving dots): it detects *some* stars but not all,
  false-positives on the **door frame** (a straight edge → exactly what the determinant-of-Hessian test rejects), and **misses
  the dim movers**. The starfield-tuned band-pass + SNR doesn't transfer to bright/cluttered
  scenes. Action: **trim a short clip as a checked-in test fixture** (full captures are multi-GB).
- **Temporal differencing to surface dim movers** — but **not** naive pairwise. *Finding (real
  capture, `framediff` tool):* the magnitude of successive-frame differences was **harder to see
  than the raw** — a dim, slowly-moving dot nearly overlaps itself between 0.2 s frames, so it
  largely **self-cancels** while √2× noise fills the frame. Pairwise diff only helps *fast*
  movers. Better for slow/dim ones: (a) **longer baseline** — diff against frame `i−K` (seconds
  back) so the mover separates into two distinct dots; (b) **temporal-median background
  subtraction** — median over a window removes static stars/clutter, and a moving dot isn't in
  the median at its current spot so it pops without self-cancellation (the standard technique).
  Then run the determinant-of-Hessian blob test on the result. (We already frame-diff for the `moving` flag — this
  is about using it to *find*, with the right baseline.)
  - **Update — treated as a dead end.** The whole moving-target / temporal angle is a red herring:
    (i) the goal is to *track*, which holds the target ~fixed, so **fixed-source detection is the
    job**; and (ii) background subtraction doesn't even null bright *static* clutter — shot-noise
    variance ∝ √brightness, so |Δ| stays large wherever the scene is bright (the IR-blown tree
    lights up a difference image despite being static). So: lean on fixed-source detection
    (determinant of the Hessian) and clear static clutter via the backend's **prediction /
    proximity gating once locked**, not temporal subtraction. (`framediff --ema` stays only as a
    standalone exploration tool; it is *not* in the detect path.)
- **False positives are OK for tracking** (so masking isn't worth it): as long as detection also
  gets the *real* targets, the backend's gated selection picks the right one near the prediction.
  FP-cleanliness matters mainly for **plate-solving**. And when the mount is **not slewing**,
  frame-differencing already cancels static clutter (door frame, tree) for free.
- **Match the PSF to a few pixels (slight defocus) + matched low-pass.** A focused star ≈ 1px —
  the *same spatial frequency* as read/hot-pixel noise — so no filter can separate them. Slightly
  defocus the **guide** cam so the PSF spans ~2–3px (Nyquist), then **low-pass / matched-filter**
  at that scale (small Gaussian convolution): it passes the blob while cutting 1px noise → higher
  detection SNR, and it enables **sub-pixel centroiding** (tighter tracking). Sweet spot ~2–3px
  FWHM (over-defocus spreads the energy thin and lowers peak SNR). Keep the **main** cam focused
  (resolve/record). The sim already uses a ~3px PSF; the real guide cam is likely undersampled,
  which partly explains the poor real-scene detection above. (Names: **matched filtering** /
  PSF-matched filter — relatives are the *Laplacian of Gaussian* and *Difference of Gaussians*;
  the optics practice is **defocused photometry** and keeping the PSF **Nyquist / critically
  sampled** for centroiding.)
- **Determinant-of-Hessian blob detector** — *built and the default*: `--detector doh` on both
  `detect` and `backend` (band-pass kept via `--detector bandpass`), single scale via `--doh-sigma`
  (0 ⇒ `--psf-px`), in `detect.det_of_hessian`. Convolve with Gaussian 2nd-derivative filters
  (Lxx, Lyy, Lxy) and take the determinant Lxx·Lyy − Lxy². Peaks are round blobs (stars, a laser
  dot); edges/lines (telephone wires) score ~0 because one principal curvature vanishes. Three
  refinements made it actually work on the real shed-door capture:
  - **sqrt-linearized**: return `sqrt(max(DoH, 0))` (geometric mean of the principal curvatures).
    The raw determinant scales as contrast², which crushed faint stars to 1–2 detections; the root
    is linear in contrast (faint-star sensitivity) and still vanishes on edges.
  - **density cap** (`--tile-grid`, `--per-tile`): keep ≤ per_tile blobs per grid tile so a dense
    bright region (the IR-blown tree) can't eat the whole `--max-candidates` budget and starve real
    targets elsewhere. Limits target *density*, not just the total.
  - **fast peak-finding**: a *strided* max-pool with `return_indices` (one max + location per
    `2·suppress_radius+1` tile, O(N) single pass) replaced a stride-1 NMS + a Python loop over
    ~25k local maxima that cost ~3 s/frame. Boundary-straddling duplicates merge in the per-peak
    `taken` mask. detect went ~3100 ms → ~80 ms/frame (~12 fps). [open: small scale-space?]
- **Candidate pipeline: PSF-scale band-pass → determinant of the Hessian.** Band-pass matched to
  the PSF (low-pass to cut the highest frequency = 1px noise; high-pass to cut frequencies below
  the PSF = background / large structure), then the determinant of the Hessian to keep round blobs
  and reject lines/edges. **But the determinant of the Hessian already bakes the band-pass in** —
  it's computed by convolving with *Gaussian second-derivative* filters, so the Gaussian's width
  is the scale + the noise low-pass, and differentiating kills the flat/slow background (the
  high-pass). So a separate band-pass is largely redundant; **determinant of the Hessian at the PSF
  scale likely does the whole job**. (If a cheap explicit band-pass is wanted, a *Difference of
  Gaussians* — narrow blur minus wide blur — approximates a *Laplacian of Gaussian*; our current
  box-blur background-subtract is a crude cousin.) Pairs naturally with the slightly-defocused,
  PSF-matched guide cam above.
- **Track-aware target selection (backend).** Once locked, weight targetness by proximity to the
  *expected* target position and reject far false matches. This is a decision-stage job and the
  backend already has the prediction (controller pos + vel), so do it there — **no detect coupling**,
  detect stays a pure file→file perception stage (live == replay). The backend extrapolates its
  last `{pos, t, vel}` to the detection's frame timestamp and picks/weights among detect's
  candidates. (Mostly the controller's existing gated association, generalized to soft weighting.)
- **Detect-side ROI gating (deferred perf optimization).** The *only* extra benefit of pushing the
  gate into detect is not running the detector over the whole frame — worth it only once detection
  is expensive (e.g. the determinant of the Hessian over a big main-cam frame). When profiling says so, add a **backend→detect
  kinematic-hint feedback path** `{pos, timestamp, velocity}`; detect extrapolates to the new
  frame's own capture timestamp and searches only that region. Couples the stages, so don't do it
  before it pays for itself.
  - *Caveats for later (the split isn't permanent):* (a) once targets are **resolved/imaged**
    rather than points, the detector carries **identity/appearance** cues worth fusing into
    selection, not just position; (b) the gate is **time-dependent** — it should grow with elapsed
    time since the last good fix as positional uncertainty accumulates, not a fixed radius.
- **Distractor rejection** (a dim target flying past a bright star): when locked, **don't select
  by brightness** — a bright star near the dim mover must not steal the lock. Score by proximity
  to the prediction + **velocity-consistency** (the star is ~sidereal/stationary while the target
  moves along its predicted track), and by brightness/appearance *similarity* once the target is
  characterized. If genuinely ambiguous through the conjunction, **coast on the prediction** and
  re-acquire on the far side.

### Guiding / control
- **Scale-invariant tuning**: the PI *loop gain* is already pixel-scale-invariant (`rad_per_px`
  cancels between command and plant), but the px-denominated tunables (gate, dead-zone,
  velocity) are not — express those in **angular units (arcsec)** so a tuning transfers across
  cameras/FOVs.
- **Live boresight-offset nudge = the calibration.** The loop holds the target at
  `frame_center + boresight_offset`; let the operator nudge that offset (gamepad/keys) *while
  tracking* until the target sits where they want (e.g. centered in the main cam). That nudge,
  persisted, *is* the boresight calibration — no separate ritual — and it can be redone on the fly
  to absorb **mirror flop**. (Supersedes the milestone-7 boresight step.)
- **Track on the main (narrow) camera** for finer guiding: a **new main-cam detect process** using
  a **centroid** algorithm (the target is resolved/bright there, not a peak-blob); the controller
  takes the main-cam error when locked on it. **Keep the guide-cam lock alive as a fallback** so we
  re-acquire when the narrow field loses it. Needs the scale-invariant-tuning item above (very
  different plate scale). A guide→main handoff, with the guide as the safety net.

### World model as a process (very stretch)
- Pull the world model out of the sim camera into its **own process**. It emits, per target, a
  **piecewise-linear path in world *direction* space** over time — sampled **ENU unit vectors**
  (renormalized-lerp between samples, which avoids the alt/az pole singularity), plus magnitude /
  identity. The sim camera and the GUI both **interpolate by timestamp** to place targets
  smoothly at any instant. This is the satellite precompute-and-interpolate already in `skysim`,
  generalized to all targets and exposed as just another file in the pipeline.
- Payoffs: decouples the heavy astrometry (compute once → many consumers); becomes the **single
  source for GUI world-model overlays** (the circle/line convention — satellites, aircraft,
  landmarks); serves both **sim** (render the truth) and **real** (overlay predictions on the live
  guide cam); and is the natural **bridge to the main AstroLock world model**, which already
  computes these positions and could emit the same track file.

### Pipeline / I/O
- **mmap the SER path for zero-copy into torch.** *Reader: built* — `SerReader(use_mmap=True)` (the
  default) returns a per-frame `np.memmap` view, with automatic seek+read fallback on any mmap error
  (`use_mmap=False` forces it off). Reader (simplest — **map per frame**):
  `torch.from_numpy(np.memmap(path, np.uint16, 'r', offset=178 + i*frame_bytes, shape=(h, w)))`.
  `np.memmap` handles the offset-alignment rule for you (raw `mmap`'s `offset=` must be a multiple of
  `ALLOCATIONGRANULARITY` = 64 KB on Windows / 4 KB on Linux — it maps from the page boundary below
  and views at your offset). **Lifetime is automatic**: `torch.from_numpy` retains the memmap as the
  tensor's base, so the mapping unmaps when the sole tensor is GC'd — no bookkeeping (wrap it if you
  want a deterministic `close()`). Cost is one map/unmap per frame — negligible below hundreds of fps.
  Read-only is fine — detection makes new tensors anyway. For our large frames this also beats
  `read()`, which memcpys the whole page-cache frame into a user buffer; mmap hands the pages straight
  to torch. (Alternatives: `torch.frombuffer` on a stdlib `mmap` opened `access=ACCESS_READ`, no
  numpy; or map the whole file once and index, if per-frame syscalls ever bite.) Writer (ZWO→SER): pre-size fixed segments (we already roll at 300
  frames → known size), mmap `'r+'`, and let the SDK DMA straight into the frame slot, dropping a copy
  on capture too.
- **Growing file** — *moot for the per-frame reader* (each frame is an already-complete range, never
  past EOF); this only matters for the whole-file-map variant and the **writer pre-sizing slots to
  DMA into**. There: **pre-sized segments, remap only at rollover.** We already roll at 300-frame
  segments, so on open `truncate()` each segment to its full final size — a **sparse** file (zeros,
  no disk until written, on NTFS/ext4) — and map the *whole* segment up front. Growth *within* a
  segment then needs **no remap**: the writer fills pre-mapped slots, the sidecar gates readable
  frames. Remap only when opening the next segment (once per 300 frames). Don't "map huge" past EOF:
  on POSIX touching unbacked pages is **SIGBUS**; on Windows a larger map *extends the file* to that
  size instead (read-only maps are capped at the file size). Either way **pre-sizing is what makes a
  big map legal**. And there's no portable *in-place* grow — `mremap` is Linux-only, and
  `mmap.resize()` can move the base pointer and **invalidate exported numpy/torch views** — so
  **remap, don't resize**: create a fresh mapping, keep the old one alive until its tensors retire,
  then drop it (only at rollover).
- **Writer with rolling *disabled* (one unbounded file).** No known final size, so pre-sizing a
  segment doesn't apply. Two options: (a) **simplest — keep the writer on plain append-`write()`**
  (extends naturally, no max size, one DMA→cache copy; reserve mmap for the readers, where zero-copy
  across consumers is the big win — *this is the default*); (b) **mmap-write via chunked pre-grow** —
  `truncate` forward by a chunk (~1 GB, sparse), map it, write frames in, remap when crossing a chunk,
  and `truncate` back to the true size (`header + n·frame_bytes`) on close to trim the zero tail. The
  writer may `resize()`/remap freely (it exports no views; readers map independently). Or (c)
  **per-frame grow — the mirror of the per-frame reader, and the cleanest:** each frame
  `os.ftruncate(fd, header + (N+1)*frame_bytes)` (sparse-extend by one slot), map that frame's
  page-aligned region `ACCESS_WRITE`, have the SDK DMA into it, then **release the map before the next
  frame's truncate**. No rolling, no chunk bookkeeping, no zero tail to trim. **Windows gotcha:** you
  can't resize a file that has a *live* mapping (`ftruncate` fails) — the release-before-truncate order
  dodges it (POSIX is lenient but the same code works). Net rule: **mmap the readers always; mmap the
  writer only if the capture-side copy profiles hot.**
- **ZWO buffer path (for the writer DMA-into-slot).** `zwoasi`'s `capture_video_frame` →
  `_get_video_data` has the SDK write *in place* via `cbuf = (c_char*n).from_buffer(buf)` +
  `ASIGetVideoData(id, cbuf, sz, timeout)`, then returns a zero-copy `np.frombuffer` view. It
  **hard-gates `buf` on `isinstance(bytearray)`**, so to DMA straight into the SER map, bypass the
  method and call the SDK directly: `cbuf = (ctypes.c_char*frame_bytes).from_buffer(mm_slot, off)`
  (`from_buffer` accepts an mmap/memoryview) then `zwolib.ASIGetVideoData(id, cbuf, frame_bytes,
  timeout)` — the frame lands in the page-cache-backed `.ser`, no `write()`. Free win regardless:
  pass a *persistent* `buffer_` so it stops allocating a fresh `bytearray` per frame.
- **What this actually saves (don't oversell it).** `ASIGetVideoData(id, unsigned char* buf, size,
  waitms)` takes a fresh pointer each call → it's a **copy-out**: the SDK `memcpy`s from its own
  internal USB-transfer buffers into your pointer (a true DMA-into-user-buffer API would make you
  *register* buffers up front; ASI has none). So (a) **your buffer needs no DMA alignment** — it's a
  memcpy target, alignment-agnostic; the real DMA-alignment lives in the kernel USB stack / SDK
  internal buffers you never touch; and (b) the mmap trick removes the **second userspace copy** (the
  `write()` to disk), not the hardware DMA — modest (one ~16 MB/frame copy), not "zero-copy."
- **Demand-zeroing makes the writer-mmap ~a wash (the real verdict).** Extending a file gives
  **demand-zeroed** pages (a security guarantee — sparse hole → kernel faults in a zero-filled page;
  no disk until real data is written). But for a slot you then fully overwrite, the kernel zeroes the
  page *and* the SDK copies over it, so the copy you "saved" is paid back as a zero-fill. The
  asymmetry is *full-page vs partial*, not write-vs-mmap per se: a **full-page `write()`** hands the
  kernel the whole page's bytes at once, so `write_begin` fills it directly and **skips the zero**
  (ext4/iomap: full-folio write over a hole isn't zeroed — a *partial* write still is, so a 1-byte
  write zero-fills too); an **mmap store can't skip it** — the page is materialized at *fault time*,
  before the kernel knows you'll overwrite it, so it's zeroed then copied over. Our ~16 MB frames are
  thousands of full pages (only a trailing partial page pays), so `write()` = 2 copies + ~no
  zero-fill vs mmap = 1 copy + per-page zero-fill ≈ same bandwidth. Same whether you grow per-frame or
  pre-size (pre-sized slots are still sparse → still zeroed on first touch). No portable escape — `MAP_UNINITIALIZED` is
  anonymous-only/kernel-flagged, `fallocate` still reads as zero. **Verdict: mmap is the clear win on
  reads (faults in real data, no zero-fill); on writes it's marginal — keep `write()` unless it
  profiles hot.**
- **The page cache becomes the IPC, almost free.** With cam/detect/gui as separate processes on one
  machine, cam `write()`s a frame and detect/gui memmap the *same* pages — no re-read from disk; the
  `.ser` effectively acts as shared memory (the sidecar already gates "frame complete" for
  coherence). This is the cheaper cousin of the shared-memory ring buffer parked below.
- **Caveat — CPU-only.** Zero-copy helps CPU-side detection and cross-process sharing; GPU upload
  still copies and mmap'd pages can't be pinned, so if the goal is feeding the GPU hard, pinned
  staging buffers matter more than mmap. [open: is the target CPU detection throughput, or GPU
  feeding?]

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
