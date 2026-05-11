
from pathlib import Path
import json

def save_json(data, directory: Path, filename: str) -> Path:
    path = directory / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path