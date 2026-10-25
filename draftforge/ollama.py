"""Thin Ollama client: install, start the server, pull models, generate."""
from __future__ import annotations
import json
import re
import os
import shutil
import subprocess
import threading
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from . import interrupt
from .util import LOG, human_bytes

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

_WIN_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
    Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
    Path("C:/Program Files/Ollama/ollama.exe"),
]


class OllamaError(RuntimeError):
    pass


def find_binary() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    for c in _WIN_CANDIDATES:
        try:
            if c and c.exists():
                return str(c)
        except OSError:
            continue
    return None


OLLAMA_SETUP_URL = "https://ollama.com/download/OllamaSetup.exe"


def _watch_progress(path: Path, total: int, label: str, stop) -> None:
    """Log download progress every 30 s."""
    t0 = time.time()
    start = path.stat().st_size if path.exists() else 0
    while not stop.wait(30):
        try:
            got = path.stat().st_size
        except OSError:
            continue
        rate = (got - start) / max(1e-6, time.time() - t0)
        eta = ((total - got) / rate) if (total and rate > 0) else 0
        LOG.info("  %s %s%s  %s/s%s", label, human_bytes(got),
                 (" / " + human_bytes(total)) if total else "",
                 human_bytes(rate),
                 ("  eta " + _hms(eta)) if eta else "")


def _hms(s: float) -> str:
    s = int(s)
    if s < 3600:
        return "%dm" % (s // 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


def _remote_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "draftforge"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _download(url: str, dest: Path, label: str = "download",
              attempts: int = 20) -> bool:
    """Download a file, resuming a partial `.part` across attempts and runs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    total = _remote_size(url)
    if total:
        LOG.info("  %s is %s", label, human_bytes(total))

    curl = shutil.which("curl")
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total and have >= total:
            break
        if interrupt.stopping():
            LOG.info("  %s paused at %s - resumes on the next run", label,
                     human_bytes(have))
            return False
        if have:
            LOG.info("  resuming %s at %s%s", label, human_bytes(have),
                     (" / " + human_bytes(total)) if total else "")

        try:
            if curl:
                # curl resumes with -C -; a watcher thread logs progress.
                stop = threading.Event()
                threading.Thread(target=_watch_progress,
                                 args=(tmp, total, label, stop),
                                 daemon=True).start()
                try:
                    subprocess.run([curl, "-L", "-C", "-", "--retry", "3",
                                    "--connect-timeout", "60",
                                    "--no-progress-meter", "-o", str(tmp), url],
                                   check=False, timeout=7200)
                finally:
                    stop.set()
            else:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "draftforge",
                                  **({"Range": "bytes=%d-" % have} if have else {})})
                with urllib.request.urlopen(req, timeout=120) as r:
                    mode = "ab" if have and r.status == 206 else "wb"
                    got = have if mode == "ab" else 0
                    last = time.time()
                    with open(tmp, mode) as f:
                        while True:
                            if interrupt.stopping():
                                break
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            got += len(chunk)
                            if time.time() - last > 10:
                                last = time.time()
                                LOG.info("  %s %s%s", label, human_bytes(got),
                                         (" / %s (%.0f%%)" % (human_bytes(total),
                                          100.0 * got / total)) if total else "")
        except Exception as e:
            LOG.warning("  %s attempt %d failed: %s", label, attempt, e)

        now = tmp.stat().st_size if tmp.exists() else 0
        if total and now >= total:
            break
        if now <= have:
            LOG.warning("  no progress on attempt %d - retrying", attempt)
            if interrupt.sleep(5):
                return False

    final = tmp.stat().st_size if tmp.exists() else 0
    if not final or (total and final < total):
        level = LOG.info if interrupt.stopping() else LOG.error
        level("  %s incomplete (%s of %s) - re-run to continue", label,
              human_bytes(final), human_bytes(total) if total else "?")
        return False
    tmp.replace(dest)
    LOG.info("  %s complete: %s", label, human_bytes(dest.stat().st_size))
    return True


def _install_windows_direct() -> bool:
    from .paths import P
    setup = P.downloads / "OllamaSetup.exe"
    if not setup.exists() or setup.stat().st_size < 500_000_000:
        LOG.info("downloading the Ollama installer (~1.6 GB, resumable)")
        if not _download(OLLAMA_SETUP_URL, setup, "OllamaSetup.exe"):
            return False
    LOG.info("running the Ollama installer silently")
    for flags in (["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], ["/S"]):
        try:
            subprocess.run([str(setup), *flags], check=False, timeout=900)
        except Exception as e:
            LOG.warning("installer (%s) failed: %s", " ".join(flags), e)
            continue
        if find_binary():
            return True
    return bool(find_binary())


def _winget_busy() -> bool:
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True,
                             timeout=20).stdout.lower()
        return "winget.exe" in out
    except Exception:
        return False


def _install_via_winget() -> bool:
    if not shutil.which("winget"):
        return False
    if _winget_busy():
        LOG.error("another winget install is already running - wait for it to "
                  "finish, then re-run. (winget allows only one at a time.)")
        return False
    LOG.info("falling back to winget")
    try:
        subprocess.run(["winget", "install", "--id", "Ollama.Ollama", "-e",
                        "--source", "winget", "--silent",
                        "--accept-package-agreements",
                        "--accept-source-agreements"], check=False, timeout=1800)
    except Exception as e:
        LOG.warning("winget install failed: %s", e)
    return bool(find_binary())


def install() -> Optional[str]:
    """Install Ollama if missing; returns the binary path."""
    exe = find_binary()
    if exe:
        LOG.info("ollama already installed at %s - skipping install", exe)
        return exe

    if os.name == "nt":
        # Direct download first: winget can block silently on a global lock.
        if _install_windows_direct() or _install_via_winget():
            pass
    elif sys.platform == "darwin":
        if not shutil.which("brew"):
            LOG.error("Homebrew not found - install Ollama from "
                      "https://ollama.com/download")
            return None
        LOG.info("installing Ollama via brew")
        subprocess.run(["brew", "install", "--cask", "ollama"], check=False,
                       timeout=1800)
    else:
        LOG.error("auto-install not supported on this platform. Run:\n"
                  "  curl -fsSL https://ollama.com/install.sh | sh")
        return None

    exe = find_binary()
    if exe:
        LOG.info("Ollama installed: %s", exe)
    else:
        LOG.error("Ollama installed but the binary was not found - open a new "
                  "terminal and re-run, or install from https://ollama.com/download")
    return exe


def _get(path: str, timeout: int = 10):
    req = urllib.request.Request(HOST + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, payload: dict, timeout: int = 600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_stream(path: str, payload: dict, timeout: int = 3600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue


def is_up() -> bool:
    try:
        _get("/api/tags", timeout=3)
        return True
    except Exception:
        return False


def ensure_server(wait_s: int = 90, auto_install: bool = True) -> None:
    """Install Ollama if missing, then start `ollama serve` if not listening."""
    if is_up():
        LOG.info("ollama already running at %s - nothing to install or start",
                 HOST)
        return
    exe = find_binary()
    if exe:
        LOG.info("ollama found at %s (not running) - starting it", exe)
    elif auto_install:
        exe = install()
    if not exe:
        raise OllamaError(
            "ollama not found. Install from https://ollama.com/download "
            "then re-run. (Nothing else in the pipeline needs it until the "
            "models stage.)")
    LOG.info("starting ollama server (%s)", exe)
    flags = 0
    if os.name == "nt":
        # Own process group, so Ctrl+C here does not kill the server.
        flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    env = dict(os.environ)
    env.setdefault("OLLAMA_NUM_PARALLEL", "6")
    env.setdefault("OLLAMA_KEEP_ALIVE", "30m")
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=flags, env=env)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if is_up():
            LOG.info("ollama server up after %.1fs", time.time() - t0)
            return
        if interrupt.sleep(1.5):
            raise interrupt.Interrupted("stopped while waiting for ollama")
    raise OllamaError("ollama server did not come up within %ds" % wait_s)


def list_models() -> list[str]:
    try:
        tags = _get("/api/tags")
    except Exception as e:
        raise OllamaError("could not list ollama models: " + str(e))
    out = []
    for m in tags.get("models", []):
        name = m.get("name") or m.get("model") or ""
        if name:
            out.append(name)
            if ":" in name:
                out.append(name.split(":")[0])
    return out


def ensure_model(tag: str, auto_install: bool = True) -> None:
    """Pull the model if missing; Ollama resumes partial pulls."""
    ensure_server(auto_install=auto_install)
    have = list_models()
    if tag in have or tag.replace(":latest", "") in have:
        LOG.info("ollama model present: %s", tag)
        return
    LOG.info("pulling ollama model %s (resumable, this can take a while)", tag)
    last = 0.0
    for ev in _post_stream("/api/pull", {"model": tag, "stream": True}):
        if "error" in ev:
            raise OllamaError("pull failed: " + str(ev["error"]))
        interrupt.check("pull stopped - it resumes on the next run")
        total, done = ev.get("total"), ev.get("completed")
        if total and done and time.time() - last > 5:
            last = time.time()
            LOG.info("  %s  %s / %s (%.1f%%)", ev.get("status", ""),
                     human_bytes(done), human_bytes(total), 100.0 * done / total)
    if tag not in list_models():
        raise OllamaError("model %s still missing after pull" % tag)
    LOG.info("pulled %s", tag)


def unload_all() -> None:
    """Evict every model Ollama holds in VRAM (keep-alive is 30 min)."""
    if not is_up():
        return
    try:
        running = _get("/api/ps").get("models") or []
    except Exception as e:
        LOG.debug("could not list loaded ollama models: %s", e)
        return
    for m in running:
        name = m.get("name") or m.get("model")
        try:
            _post("/api/generate", {"model": name, "keep_alive": 0}, timeout=60)
            LOG.info("unloaded %s from ollama to free GPU memory", name)
        except Exception as e:
            LOG.warning("could not unload %s from ollama (%s) - training may "
                        "run short of GPU memory", name, e)


def detect_parallel_slots(default: int = 1) -> int:
    """Request slots llama-server really has (`-np`); Qwen3.5 gets 1 because
    of its vision projector, and extra client threads only queue."""
    if os.name != "nt":
        return default
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\")"
             ".CommandLine"],
            capture_output=True, text=True, timeout=60)
        m = re.search(r"-np\s+(\d+)", r.stdout or "")
        if m:
            return max(1, int(m.group(1)))
    except Exception as e:
        LOG.debug("slot detection failed: %s", e)
    return default


def generate(model: str, prompt: str, *, max_tokens: int, temperature: float,
             top_p: float, ctx: int, timeout: int, seed: int | None = None,
             think: bool = False) -> dict:
    """One completion. Keep `think` off: Qwen3.5 otherwise spends the whole
    budget reasoning and returns an empty response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": bool(think),
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": ctx,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    try:
        r = _post("/api/generate", payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        # older models reject `think`
        if e.code != 400:
            raise
        payload.pop("think", None)
        r = _post("/api/generate", payload, timeout=timeout)

    text = r.get("response") or ""
    if not text.strip():
        text = (r.get("thinking") or "").strip()
    return {
        "response": text,
        "thinking_only": not (r.get("response") or "").strip(),
        "eval_count": r.get("eval_count", 0),
        "prompt_eval_count": r.get("prompt_eval_count", 0),
        "total_duration_ns": r.get("total_duration", 0),
    }
