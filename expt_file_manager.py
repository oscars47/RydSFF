## file manager for the experiments with quera
# assigns unique indentifier for h_ls, x, ev_params, t_plateau, t_ramp, seq, n_shots, gate_params, and timestamp

from __future__ import annotations

import json, os, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

# ---------------- helpers ----------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def make_uid(payload: Dict[str, Any], digest_size: int = 12) -> str:
    can = canonical_json(payload).encode("utf-8")
    return hashlib.blake2b(can, digest_size=digest_size).hexdigest()

def atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)

# ---------------- store ------------------

class ExptStore:
    """
    Directory layout:

        root/
          index.json
          combos/
            <uid>.json

    Example call: uid, added = store.add(params, timestamp=timestamp)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.combo_dir = self.root / "combos"
        self.index_path = self.root / "index.json"
        self.data_dir = self.root / "data"
        self.task_dir = self.root / "tasks"
        self.results_dir = self.root / "results"

        
        self.root.mkdir(parents=True, exist_ok=True)
        self.combo_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = {
                "created_at": iso_now(),
                "records": {}
            }
            atomic_write(self.index_path, self.index)

    def add(
        self,
        payload: Dict[str, Any],
        *,
        timestamp: Any | None = None,
        ts_key: str = "timestamp"
    ) -> Tuple[str, bool]:
        """
        Insert 'timestamp' into the payload and use the WHOLE payload to build the UID.

        Parameters
        ----------
        payload : dict
            Your parameters (must be JSON-serializable).
        timestamp : Any
            If None, we'll use iso_now(). Otherwise whatever you pass (string, float, etc.) is stored & hashed.
        ts_key : str
            Field name to store the timestamp under.
        """
        # clone & inject timestamp
        payload = dict(payload)
        payload[ts_key] = timestamp if timestamp is not None else iso_now()

        # UID now depends on timestamp
        uid = make_uid(payload)

        if uid in self.index["records"]:
            return uid, False

        record = {
            "id": uid,
            "created_at": payload[ts_key],
            "payload": payload
        }
        combo_path = self.combo_dir / f"{uid}.json"
        atomic_write(combo_path, record)

        self.index["records"][uid] = {
            "id": uid,
            "created_at": payload[ts_key],
            "path": str(combo_path.relative_to(self.root)),
        }
        atomic_write(self.index_path, self.index)

        return uid, True

    def get(self, uid: str) -> Dict[str, Any] | None:
        meta = self.index["records"].get(uid)
        if not meta:
            return None 
        with (self.root / meta["path"]).open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_ids(self):
        return list(self.index["records"].keys())

    def list_all(self):
        return [self.get(uid) for uid in self.list_ids()]

    def remove(self, uid: str) -> bool:
        meta = self.index["records"].get(uid)
        if not meta:
            return False
        try:
            (self.root / meta["path"]).unlink(missing_ok=True)
        finally:
            del self.index["records"][uid]
            atomic_write(self.index_path, self.index)
        return True
    
def unique_filename(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}-{counter}{ext}"
        counter += 1
    print(f"File {filename} exists, using new filename {new_filename}")
    return new_filename


if __name__ == "__main__":
    ## sample usage
    store = ExptStore("root_test")
    params = {
        "h_ls": [0.1, 0.2],
        "x": [(0,1), (1,0)],
        "ev_params": {"Omega": 10},
        "t_plateau": 2.0,
        "t_ramp": 0.4,
        "seq": [0, 1, 1],
        "n_shots": 1000,
        "gate_params": {"phi": 0.7},
    }

    timestamp = 123456
    uid, added = store.add(params, timestamp=timestamp)
    print(uid, added)

    # if you skip timestamp, it autogenerates one and still hashes it
    uid2, added2 = store.add(params)  # different timestamp -> different UID
    print(uid2, added2)

    # duplicate, returns False
    uid2, added2 = store.add(params, timestamp=timestamp) 
    print(uid2, added2)
