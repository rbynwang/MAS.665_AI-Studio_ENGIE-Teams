"""Compact read-only home data, so 200 homes fit one Jetson process. Nothing here changes a number.

A year of one home as Python dicts takes about 31 MB, so 200 homes took about 6.2 of the fleet's 6.8 GB (measured 6 Nov 2026).
A pack keeps the same year in one memory-mapped file: the operating system loads only the pages a home touches, and it can
drop them again. Any process on the machine shares them.

  python -m flexgrid_core.lean pack --out ~/flexgrid-v3/pack        # once per dataset, after the data changes
  lean.install(pack_dir)                                            # at a daemon's start, before any Replay is made

install() fills replay._CACHE and hvac._CACHE, the caches those modules already keep, so neither file changes and every
Replay works as before. rows[i] gives the same dict the CSV gives. Each number sits in a 4-byte integer, not a float32:
float32 would turn 0.257 into 0.25699999928. The dataset has at most 3 decimals, and an integer over 1000 divides back to
exactly the double the CSV parser gives, because both round the same exact fraction. build() checks every value.

Rows a home has read stay as dicts in a small cache: a planning step reads the same rows many times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config, hvac, replay as replay_mod

VERSION = 1
SCALE = 1000.0            # 3 decimals, the most the dataset has
KEEP_ROWS = 640           # rows a home keeps as dicts: 6.7 days of steps, about 0.4 MB
HVAC_COLS = ("cool", "cop", "tin")
HVAC_FLAGS = ("occ", "mode")


class LeanRows:
    """Behaves as the list of row dicts replay._load returns, for reading. A row is made when it is first asked for."""
    __slots__ = ("m", "keys", "ts", "n", "keep", "hits", "made")

    def __init__(self, m: np.ndarray, keys: tuple[str, ...], ts: list[str]):
        self.m, self.keys, self.ts, self.n = m, keys, ts, len(ts)
        self.keep: dict[int, dict[str, Any]] = {}
        self.hits = self.made = 0

    def __len__(self) -> int:
        return self.n

    def _make(self, i: int) -> dict[str, Any]:
        r = dict(zip(self.keys, (self.m[i] / SCALE).tolist()))
        r["ts_local"] = self.ts[i]
        return r

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self.n))]
        if i < 0:
            i += self.n
        keep = self.keep
        r = keep.get(i)
        if r is not None:
            self.hits += 1
            return r
        if not 0 <= i < self.n:
            raise IndexError(i)
        r = self._make(i)
        self.made += 1
        if len(keep) >= KEEP_ROWS:
            try:
                keep.pop(next(iter(keep)), None)              # the oldest row read
            except (RuntimeError, StopIteration):             # another thread changed it: the next read tries again
                pass
        keep[i] = r
        return r

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for a in range(0, self.n, 1024):                      # a pass over the year doesn't push out what planning keeps
            for i, vals in enumerate((self.m[a:a + 1024] / SCALE).tolist(), a):
                r = dict(zip(self.keys, vals))
                r["ts_local"] = self.ts[i]
                yield r


def _pack_dir(root: str | os.PathLike | None) -> Path:
    base = Path(os.path.expanduser(str(root or os.environ.get("FLEXGRID_PACK_DIR") or config.REPO_DIR / "pack")))
    return base / config.DATASET


def _map(path: Path) -> np.ndarray:
    """A plain array over the mapped file: np.memmap is slower to index, and this holds the same mapping."""
    return np.asarray(np.load(path, mmap_mode="r"))


def _home_files(profile: str) -> list[Path]:
    return [config.data_path(f"{profile}.csv.gz"), config.data_path(f"{profile}_ev.csv"), config.data_path(f"{profile}_hvac.csv.gz")]


def _stamp(profiles: list[str]) -> str:
    """Changes when any home's data changes: each source file's bytes. A size would miss a corrected value, because a digit changed in a
    csv leaves the size as it was. 150 to 500 MB of files, so about a second: cheap enough to check at every start."""
    h = hashlib.sha1()
    for p in profiles:
        for f in _home_files(p):
            h.update(f"{f.name}:".encode() + (f.read_bytes() if f.exists() else b"missing") + b";")
    return h.hexdigest()


def _roster() -> list[dict[str, Any]]:
    return json.loads(config.data_path("fleet_v2.json").read_text())["homes"]


# ---------------------------------------------------------------------------------------------- writing a pack
def build(root: str | os.PathLike | None = None, log=print) -> dict[str, Any]:
    """Writes the active dataset's pack. Reads each home through replay._load and hvac's reader, so it holds what they give,
    then reads every value back. Refuses a value that doesn't come back exact."""
    t0 = time.time()
    out = _pack_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    profiles = [h["profile"] for h in _roster()]
    n_home = len(profiles)
    rows0, _, _ = replay_mod._load(profiles[0])
    keys = tuple(k for k in rows0[0] if k != "ts_local")
    ts = [r["ts_local"] for r in rows0]
    n = len(ts)
    m = np.lib.format.open_memmap(out / "rows.tmp.npy", mode="w+", dtype=np.int32, shape=(n_home, n, len(keys)))
    hv = np.lib.format.open_memmap(out / "hvac.tmp.npy", mode="w+", dtype=np.float32, shape=(n_home, len(HVAC_COLS), n))
    hf = np.lib.format.open_memmap(out / "hvac_flags.tmp.npy", mode="w+", dtype=np.int8, shape=(n_home, len(HVAC_FLAGS), n))
    sessions: dict[str, list[dict[str, Any]]] = {}
    has_hvac: dict[str, bool] = {}
    for k, p in enumerate(profiles):
        rows, _, sess = replay_mod._load(p)
        if [r["ts_local"] for r in rows] != ts or tuple(x for x in rows[0] if x != "ts_local") != keys:
            raise ValueError(f"{p}: its steps or columns differ from the first home's")
        v = np.array([[r[c] for c in keys] for r in rows], dtype=np.float64)
        q = np.rint(v * SCALE)
        if np.abs(q).max() >= 2 ** 31 or not np.array_equal(q / SCALE, v):
            bad = keys[int(np.argwhere(q / SCALE != v)[0][1])] if not np.array_equal(q / SCALE, v) else "range"
            raise ValueError(f"{p}: {bad} doesn't fit 3 decimals in an integer: the pack would change a number")
        m[k] = q.astype(np.int32)
        sessions[p] = sess
        replay_mod._CACHE.pop(p, None)                        # one home at a time: this stays small
        s = hvac._series(p)
        has_hvac[p] = s is not None
        if s is not None:
            if len(s.cool) != n:
                raise ValueError(f"{p}: its cooling file has {len(s.cool)} steps, not {n}")
            hv[k, 0], hv[k, 1], hv[k, 2] = s.cool, s.cop, s.tin
            hf[k, 0], hf[k, 1] = s.occ, s.mode
        hvac._CACHE.pop(p, None)
        if (k + 1) % 50 == 0:
            log(f"   {k + 1} of {n_home} homes")
    for a in (m, hv, hf):
        a.flush()
    del m, hv, hf
    for name in ("rows", "hvac", "hvac_flags"):
        os.replace(out / f"{name}.tmp.npy", out / f"{name}.npy")
    meta = {"version": VERSION, "dataset": config.DATASET, "profiles": profiles, "keys": keys, "ts": ts, "steps": n, "scale": SCALE, "has_hvac": has_hvac,
            "sessions": sessions, "stamp": _stamp(profiles), "built": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta))
    size = sum(f.stat().st_size for f in out.iterdir())
    info = {"dataset": config.DATASET, "homes": n_home, "steps": n, "columns": list(keys), "mb": round(size / 1048576, 1), "seconds": round(time.time() - t0, 1)}
    log(f"pack written: {info}")
    return info


# ---------------------------------------------------------------------------------------------- reading a pack
def install(root: str | os.PathLike | None = None, stale_ok: bool = False) -> dict[str, Any]:
    """Fills the replay and cooling caches from the pack. It must run before any Replay is made. A pack made from other data
    stops the daemon here: the plain CSV path needs about 6 GB, so a quiet fallback would take the Jetson down."""
    d = _pack_dir(root)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"no pack at {d}: run python -m flexgrid_core.lean pack")
    meta = json.loads((d / "meta.json").read_text())
    if meta["version"] != VERSION or meta["dataset"] != config.DATASET:
        raise ValueError(f"pack at {d} is {meta['dataset']} version {meta['version']}, not {config.DATASET} version {VERSION}")
    if not stale_ok and _stamp(meta["profiles"]) != meta["stamp"]:
        raise ValueError(f"pack at {d} is stale: the data changed since {meta['built']}. Run python -m flexgrid_core.lean pack")
    m, hv, hf = _map(d / "rows.npy"), _map(d / "hvac.npy"), _map(d / "hvac_flags.npy")
    keys, ts = tuple(meta["keys"]), list(meta["ts"])
    index: dict[str, int] = {}
    for i, t in enumerate(ts):
        index.setdefault(t, i)                                # the repeated hour when clocks fall back: its first copy
    hvac_index = {t: k for k, t in enumerate(ts)}             # hvac.py's own rule: its last copy
    for k, p in enumerate(meta["profiles"]):
        rows = LeanRows(m[k], keys, ts)
        replay_mod._CACHE[p] = (rows, index, meta["sessions"][p])
        hvac._CACHE[p] = hvac._Series(hvac_index, hv[k, 0], hv[k, 1], hv[k, 2], hf[k, 0], hf[k, 1]) if meta["has_hvac"][p] else None
    return {"pack": str(d), "homes": len(meta["profiles"]), "built": meta["built"], "mb_on_disk": round(sum(f.stat().st_size for f in d.iterdir()) / 1048576, 1)}


def stats() -> dict[str, Any]:
    """How the row cache is doing, for a daemon's /health."""
    rs = [v[0] for v in replay_mod._CACHE.values() if isinstance(v[0], LeanRows)]
    return {"homes": len(rs), "rows_kept": sum(len(r.keep) for r in rs), "rows_made": sum(r.made for r in rs), "rows_reused": sum(r.hits for r in rs)}


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m flexgrid_core.lean")
    ap.add_argument("cmd", choices=["pack", "check"])
    ap.add_argument("--out", default=None, help="the pack folder (default $FLEXGRID_PACK_DIR, else <repo>/pack)")
    a = ap.parse_args()
    if a.cmd == "pack":
        build(a.out)
    else:
        print(install(a.out))


if __name__ == "__main__":
    main()
