from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np

def save_npz(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)

def load_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
