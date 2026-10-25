"""Cooperative Ctrl+C: the first press sets a flag that long loops poll to stop
at a safe point; the second quits at once."""
from __future__ import annotations
import os
import signal
import threading

STOP = threading.Event()
_hits = 0
_installed = False


class Interrupted(KeyboardInterrupt):
    """Raised by check(). A KeyboardInterrupt, so `except Exception` won't swallow it."""


def _say(msg: str) -> None:
    # a signal handler must not take logging locks
    try:
        os.write(2, msg.encode("utf-8", "replace"))
    except Exception:
        pass


def _handler(signum, frame) -> None:
    global _hits
    _hits += 1
    if _hits == 1:
        STOP.set()
        _say("\n[stopping] saving progress at the next safe point - "
             "press Ctrl+C again to quit now\n")
    else:
        _say("\n[quit] leaving immediately; the last save is intact\n")
        os._exit(130)


def install() -> None:
    """Take over Ctrl+C (and SIGTERM). Safe to call more than once."""
    global _installed
    if _installed:
        return
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass        # not the main thread, or unsupported here
    _installed = True


def stopping() -> bool:
    return STOP.is_set()


def check(where: str = "") -> None:
    """Raise Interrupted if a stop was requested."""
    if STOP.is_set():
        raise Interrupted(where or "stopped by user")


def sleep(seconds: float) -> bool:
    """Interruptible sleep. True if it was cut short by a stop request."""
    return STOP.wait(seconds)


def reset() -> None:
    """Clear a stop request (used by the self-test)."""
    global _hits
    _hits = 0
    STOP.clear()
