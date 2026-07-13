"""
Persistent settings store for AstroLock Seeker.

Named JSON files under a per-user config directory -- ``%APPDATA%\\astrolock`` on Windows,
``~/.astrolock`` elsewhere. Both the GUI (live, via the Settings tab) and the backend (at launch,
later) read/write these, so a saved set fully describes a rig: mount, cameras, optics, owned gear,
and display preferences. This module is deliberately dependency-free (no torch / GUI toolkit) so either
process can import it.

    from astrolock.seeker import settings
    settings.save('shed rig', {...})
    data = settings.load('shed rig')
    names = settings.list_settings()
"""

import json
import os
import sys


def config_dir():
    """The per-user AstroLock config directory (created on demand)."""
    if sys.platform == 'win32':
        base = os.environ.get('APPDATA') or os.path.expanduser('~')
        d = os.path.join(base, 'astrolock')
    else:
        d = os.path.join(os.path.expanduser('~'), '.astrolock')
    os.makedirs(d, exist_ok=True)
    return d


def settings_dir():
    """Where the named settings files live: <config_dir>/settings/ (created on demand)."""
    d = os.path.join(config_dir(), 'settings')
    os.makedirs(d, exist_ok=True)
    return d


def _safe_name(name):
    """Sanitise a user-typed name into a filename stem (no path traversal / odd chars)."""
    keep = ''.join(c for c in str(name) if c.isalnum() or c in ' _-().').strip()
    return keep or 'default'


def _path(name):
    return os.path.join(settings_dir(), _safe_name(name) + '.json')


def list_settings():
    """Sorted names (without .json) of the saved settings files."""
    try:
        return sorted(f[:-len('.json')] for f in os.listdir(settings_dir()) if f.endswith('.json'))
    except OSError:
        return []


def load(name):
    """The settings dict saved under `name`, or {} if it doesn't exist. A file that EXISTS but
    won't parse is reported loudly -- silently returning {} would let the GUI start on defaults
    and overwrite the user's saved calibration without a word."""
    try:
        with open(_path(name), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"[settings] could NOT load {name!r} ({e}); starting from defaults -- "
              f"saving now would overwrite it", flush=True)
        return {}


def save(name, data):
    """Write `data` (a JSON-serialisable dict) under `name`. Returns the sanitised name used."""
    safe = _safe_name(name)
    tmp = _path(safe) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _path(safe))                       # atomic swap, so a reader never sees a torn file
    return safe


def delete(name):
    """Remove the named settings file (no error if it doesn't exist)."""
    try:
        os.remove(_path(name))
    except OSError:
        pass
