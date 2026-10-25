"""Small helpers: atomic IO, hashing, formatting, logging."""
from __future__ import annotations
import hashlib, json, logging, os, shutil, sys
from datetime import datetime
from pathlib import Path

LOG = logging.getLogger("draftforge")


def setup_logging(logfile: Path, verbose: bool = False) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S"))
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s %(message)s"))
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(sh)
    root.addHandler(fh)
    for noisy in ("urllib3", "filelock", "datasets", "huggingface_hub", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        f = open(tmp, "wb")
    else:
        f = open(tmp, "w", encoding="utf-8", newline="\n")
    try:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    finally:
        f.close()
    os.replace(tmp, path)


def write_json(path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        LOG.warning("corrupt json %s - ignoring", p)
        return default


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def sha1(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def on_battery():
    """True on battery, False on mains, None if unknown (Windows only)."""
    if os.name != "nt":
        return None
    import ctypes

    class _Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
                    ("pct", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = _Status()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
    except Exception:
        return None
    return {0: True, 1: False}.get(s.ac)


def human_time(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def human_num(n: float) -> str:
    for u in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{u}".replace(".0", "")
        n /= 1000
    return f"{n:.1f}T"


def rmtree(p) -> None:
    shutil.rmtree(Path(p), ignore_errors=True)


def dir_size(p) -> int:
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
