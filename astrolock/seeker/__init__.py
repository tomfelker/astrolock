"""
AstroLock Seeker: a simple closed-loop tracker for smoothly-moving bright objects.

See astrolock_seeker.md (repo root) for the architecture/MVP design. This package is a
pipeline of small, file-driven processes:

    astrolock.seeker.cam      - per-camera capture to .ser + .frames.jsonl
    astrolock.seeker.detect   - bright/moving blob detection -> .detections.jsonl
    astrolock.seeker.backend  - orchestrator + control loop
    astrolock.seeker.gui      - Dear PyGui front end (live + playback)

plus shared library modules (ser, follower, session) imported by whichever processes
need them.
"""
