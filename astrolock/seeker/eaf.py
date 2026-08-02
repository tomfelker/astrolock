"""
ZWO EAF (Electronic Automatic Focuser) driver -- a thin veneer over the zwoeafpy ctypes
binding, mirroring cam.py's handling of the camera SDK.

The EAF is an open-loop stepper: no encoder, the firmware just counts commanded steps and
persists the count (and the max-step limit) in the device's flash. Driver tools (ASIStudio /
ASCOM) configure position and limits once for a given mechanical setup; our job is to RESPECT
them, never set them -- so this wrapper exposes only the read/move surface (position, move,
stop, is-moving, max-step, temperature) and deliberately omits EAFResetPostion / EAFSetMaxStep
/ backlash / reverse writes. Positions are absolute device steps in [0, max_step].
"""

import importlib.util
import os

_eaf = None


def _eaf_module():
    """Import zwoeafpy and load the EAF SDK library from the path we choose (ZWO_EAF_LIB or the
    ASIStudio default). Same dance as cam.py's _zwo_module, for the same reason: the module tries
    to load its DLL AT IMPORT via find_library('EAF_focuser'), which on Windows searches only
    PATH -- so the SDK dir must be findable there before the import runs (appended, never
    shadowing a system copy). add_dll_directory covers the WinAPI LoadLibrary search + the DLL's
    dependencies; if the import still didn't load it, init() from the explicit path raises loudly
    on a genuinely bad path. Cached, so the setup runs once."""
    global _eaf
    if _eaf is not None:
        return _eaf
    lib = os.getenv('ZWO_EAF_LIB') or 'C:/Program Files/ASIStudio/EAF_focuser.dll'
    sdk_dir = os.path.dirname(lib)
    if os.path.isdir(sdk_dir):
        if sdk_dir not in os.environ.get('PATH', '').split(os.pathsep):
            os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + sdk_dir
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(sdk_dir)
    import zwoeafpy
    if zwoeafpy.eaf_zwolib is None:                      # import didn't load it -> exact path
        zwoeafpy.init(lib)
    _eaf = zwoeafpy
    return _eaf


def _eaf_installed():
    """True if the zwoeafpy *module* is installed -- checked WITHOUT importing it (import has the
    load-the-DLL side effect that must wait for _eaf_module's path setup). A missing/broken DLL
    is deliberately NOT hidden here; it surfaces loudly on first real use."""
    return importlib.util.find_spec('zwoeafpy') is not None


def list_focuser_names():
    """Model names of the attached EAFs, in SDK enumeration order. Returns [] ONLY when the
    zwoeafpy module isn't installed; a broken SDK DLL propagates loudly instead of masquerading
    as 'no focusers'. None attached is not an error -- the count is just 0."""
    if not _eaf_installed():
        return []
    z = _eaf_module()
    return [z._get_focuser_property(i)['Name'] for i in range(z.get_num_focusers())]


class Focuser:
    """One open EAF. Read/move only -- see the module docstring for what's deliberately absent."""

    def __init__(self, index):
        z = _eaf_module()
        self._z = z
        n = z.get_num_focusers()
        if not 0 <= int(index) < n:              # zwoeafpy's own check leaves a half-constructed
            raise IndexError(f"no EAF at index {index} ({n} attached)")   # object that whines at GC
        self._f = z.Focuser(int(index))
        prop = self._f.get_focuser_property()
        self.name = prop['Name']
        self.max_step = self._f.get_max_step()           # the persisted, driver-configured limit

    def close(self):
        self._f.close()

    def position(self):
        """Current absolute step count."""
        return self._f.get_position()

    def move_to(self, abs_pos):
        """Command an absolute move. The caller keeps targets inside [0, max_step]; the firmware
        clamps regardless. Returns immediately -- poll is_moving() for completion."""
        self._f.move_focuser(int(abs_pos))

    def stop(self):
        self._f.stop_focuser()

    def is_moving(self):
        """(moving, moving_by_hand_controller) bools."""
        m, manual = self._f.is_moving()
        return bool(m), bool(manual)

    def temperature_c(self):
        """Probe temperature in Celsius, or None exactly while a hand-controller move is in
        progress (the SDK answers 'General error' then -- a documented transient, not a fault;
        any other error still raises)."""
        try:
            return float(self._f.get_temp())
        except self._z.ZWO_IOError as e:
            if e.error_code == 7:
                return None
            raise
