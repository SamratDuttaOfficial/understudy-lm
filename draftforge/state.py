"""Resume bookkeeping: stage status (Manifest) and saved artifacts (Registry)."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import P
from .util import LOG, now_iso, read_json, write_json


class Manifest:
    def __init__(self) -> None:
        self.d: Dict[str, Any] = read_json(P.manifest, {}) or {}
        self.d.setdefault("stages", {})
        self.d.setdefault("created", now_iso())

    def save(self) -> None:
        self.d["updated"] = now_iso()
        write_json(P.manifest, self.d)

    def entry(self, stage: str) -> dict:
        return self.d["stages"].get(stage, {})

    def is_done(self, stage: str, hash_: str) -> bool:
        e = self.entry(stage)
        return e.get("status") == "done" and e.get("hash") == hash_

    def finished_at(self, stage: str) -> float:
        """When the stage last completed, as a timestamp (0 if never)."""
        e = self.entry(stage)
        if e.get("status") != "done":
            return 0.0
        if e.get("finished_t"):
            return float(e["finished_t"])
        try:   # older entries lack finished_t
            from datetime import datetime
            return datetime.fromisoformat(e["finished"]).timestamp()
        except Exception:
            return 0.0

    def stale_reason(self, stage: str, hash_: str,
                     inputs: tuple = ()) -> Optional[str]:
        """Why a stage must run, or None if it is done and current."""
        e = self.entry(stage)
        if not e:
            return "never run"
        if e.get("status") != "done":
            return "status=" + str(e.get("status"))
        if e.get("hash") != hash_:
            return "config changed"
        mine = self.finished_at(stage)
        for up in inputs:
            if self.finished_at(up) > mine:
                return up + " re-ran"
        return None

    def mark_running(self, stage: str, hash_: str) -> None:
        prev = self.entry(stage)
        # a resumed stage keeps its elapsed time
        carry = (prev.get("elapsed_s", 0.0)
                 if prev.get("status") in ("stopped", "failed", "running") else 0.0)
        self.d["stages"][stage] = {
            "status": "running", "hash": hash_, "started": now_iso(),
            "t_start": time.time(), "elapsed_s": carry,
            "meta": prev.get("meta", {}),
        }
        self.save()

    @staticmethod
    def _elapsed(e: dict) -> float:
        """Time spent in the stage across all sessions."""
        t0 = e.get("t_start")
        return e.get("elapsed_s", 0.0) + (time.time() - t0 if t0 else 0.0)

    def mark_done(self, stage: str, hash_: str, meta: dict | None = None) -> None:
        prev = self.entry(stage)
        self.d["stages"][stage] = {
            "status": "done", "hash": hash_,
            "started": prev.get("started"), "finished": now_iso(),
            "finished_t": time.time(), "elapsed_s": self._elapsed(prev),
            "meta": meta or {},
        }
        self.save()

    def mark_failed(self, stage: str, err: str) -> None:
        e = self.d["stages"].setdefault(stage, {})
        e.update(status="failed", error=str(err)[:500], failed_at=now_iso(),
                 elapsed_s=self._elapsed(e))
        e.pop("t_start", None)
        self.save()

    def mark_stopped(self, stage: str) -> None:
        e =self.d["stages"].setdefault(stage, {})
        e.update(status="stopped", stopped_at=now_iso(),
                 elapsed_s=self._elapsed(e))
        e.pop("error", None)
        e.pop("t_start", None)
        self.save()

    def clear(self, stage: str) -> None:
        self.d["stages"].pop(stage, None)
        self.d.pop("progress::" + stage, None)
        self.save()

    # progress inside a stage
    def progress(self, stage: str) -> dict:
        return self.d.setdefault("progress::" + stage, {})

    def set_progress(self, stage: str, key: str, value: Any) -> None:
        self.progress(stage)[key] = value
        self.save()


class Registry:
    """Append-only list of timestamped artifacts (checkpoints, eval runs)."""

    def __init__(self) -> None:
        self.d: Dict[str, Any] = read_json(P.registry, {}) or {}
        self.d.setdefault("items", [])

    def save(self) -> None:
        write_json(P.registry, self.d)

    def add(self, id_: str, kind: str, path, parent: str | None = None,
            metrics: dict | None = None, config: dict | None = None,
            note: str = "") -> dict:
        item = {"id": id_, "kind": kind, "path": str(path), "parent": parent,
                "created": now_iso(), "metrics": metrics or {},
                "config": config or {}, "note": note, "active": False}
        self.d["items"] = [i for i in self.d["items"] if i["id"] != id_]
        self.d["items"].append(item)
        self.save()
        return item

    def get(self, id_: str) -> Optional[dict]:
        for i in self.d["items"]:
            if i["id"] == id_:
                return i
        return None

    def by_kind(self, kind: str) -> List[dict]:
        return sorted([i for i in self.d["items"] if i["kind"] == kind],
                      key=lambda x: x["created"])

    def latest(self, kind: str) -> Optional[dict]:
        items = self.by_kind(kind)
        return items[-1] if items else None

    def active(self, kind: str) -> Optional[dict]:
        for i in self.by_kind(kind):
            if i.get("active"):
                return i
        return self.latest(kind)

    def promote(self, id_: str) -> Optional[dict]:
        it = self.get(id_)
        if not it:
            return None
        for i in self.d["items"]:
            if i["kind"] == it["kind"]:
                i["active"] = (i["id"] == id_)
        self.save()
        LOG.info("promoted %s (kind=%s) to active", id_, it["kind"])
        return it

    def resolve(self, ref: str) -> Optional[dict]:
        """Exact id, unique substring, or a file path."""
        if not ref:
            return None
        it = self.get(ref)
        if it:
            return it
        hits = [i for i in self.d["items"] if ref in i["id"]]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            LOG.error("ambiguous ref %r -> %s", ref, [h["id"] for h in hits])
            return None
        p = Path(ref)
        if p.exists():
            return {"id": p.stem, "kind": "path", "path": str(p), "metrics": {}}
        return None

    def prune_missing(self) -> None:
        keep = [i for i in self.d["items"] if Path(i["path"]).exists()]
        if len(keep) != len(self.d["items"]):
            self.d["items"] = keep
            self.save()
