"""
Fresh TLEs from Celestrak, offline-first.

One group file (default 'stations') is downloaded per UTC day -- the date is baked into the
cache filename (the old astrolock trick), so a same-day call never touches the network and a
new day naturally refreshes. The requested satellite is picked out BY NORAD ID (names drift;
ids don't) and written to its own little 3-line file for sky_sim.

Field-first fallbacks: no network -> the newest cached day of the group; no cache at all ->
the caller's checked-in fallback file. A fetch failure is a note, never an exception.
"""

import datetime
import glob
import os
import re
import urllib.request

CELESTRAK_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle'


def tle_epoch_utc(line1):
    """UTC datetime of a TLE's epoch (line 1 cols 19-32: YYDDD.DDDDDDDD)."""
    yy = int(line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy             # NORAD convention
    day = float(line1[20:32])
    return (datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(days=day - 1.0))


def _parse_group(text):
    """{norad_id: (name, line1, line2)} from a 3-line-element group file."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out = {}
    i = 0
    while i < len(lines):
        if (lines[i].startswith('1 ') and i + 1 < len(lines)
                and lines[i + 1].startswith('2 ')):
            name, l1, l2, step = 'SAT', lines[i], lines[i + 1], 2        # bare 2-line sets
        elif (i + 2 < len(lines) and lines[i + 1].startswith('1 ')
                and lines[i + 2].startswith('2 ')):
            name, l1, l2, step = lines[i].strip(), lines[i + 1], lines[i + 2], 3
        else:
            i += 1
            continue
        m = re.match(r'^1 (\d{1,5})', l1)
        if m:
            out[int(m.group(1))] = (name, l1, l2)
        i += step
    return out


def fresh_tle_file(norad_id, cache_dir, group='stations', fallback=None, download=True,
                   timeout_s=10.0):
    """Best available TLE for ``norad_id`` as a 3-line file sky_sim can load.

    Returns {'path', 'source': 'downloaded'|'cache'|'fallback'|None, 'name',
    'epoch_utc': datetime|None, 'sig': str|None}. ``sig`` identifies the element set
    (line1+line2), so a caller can tell whether a re-check actually changed anything.
    """
    os.makedirs(cache_dir, exist_ok=True)
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    today_path = os.path.join(cache_dir, f'celestrak_{today}_{group}.txt')
    note = None
    if download and not os.path.exists(today_path):
        try:
            req = urllib.request.Request(CELESTRAK_URL.format(group=group),
                                         headers={'User-Agent': 'astrolock-seeker'})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                text = r.read().decode('utf-8', 'replace')
            if '1 ' not in text:                           # an error page, not elements
                raise ValueError('response contains no TLE lines')
            with open(today_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
        except Exception as e:                             # offline in the field: use the cache
            note = f'download failed ({e})'

    cached = sorted(glob.glob(os.path.join(cache_dir, f'celestrak_*_{group}.txt')))
    for path in reversed(cached):                          # newest day first (dates sort)
        try:
            with open(path, encoding='utf-8') as f:
                sats = _parse_group(f.read())
        except OSError:
            continue
        if norad_id in sats:
            name, l1, l2 = sats[norad_id]
            out_path = os.path.join(cache_dir, f'tle_{norad_id}.tle')
            with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(f'{name}\n{l1}\n{l2}\n')
            return {'path': out_path, 'name': name, 'epoch_utc': tle_epoch_utc(l1),
                    'sig': l1 + '|' + l2, 'note': note,
                    'source': 'downloaded' if path == today_path else 'cache'}
    if fallback and os.path.exists(fallback):
        try:
            with open(fallback, encoding='utf-8') as f:
                sats = _parse_group(f.read())
            name, l1, l2 = next(iter(sats.values())) if norad_id not in sats else sats[norad_id]
            return {'path': fallback, 'name': name, 'epoch_utc': tle_epoch_utc(l1),
                    'sig': l1 + '|' + l2, 'note': note, 'source': 'fallback'}
        except Exception:
            pass
    return {'path': fallback, 'name': None, 'epoch_utc': None, 'sig': None,
            'note': note, 'source': 'fallback' if fallback else None}
