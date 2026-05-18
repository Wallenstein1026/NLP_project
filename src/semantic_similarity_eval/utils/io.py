import csv
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_result_dir(results_dir: Path, dataset: str) -> Path:
    return ensure_dir(results_dir / dataset)


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Dict], append: bool = False) -> None:
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict) -> None:
    write_jsonl(path, [record], append=True)


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_pickle(path: Path, obj) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_pickle(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("rb") as f:
        return pickle.load(f)


def load_legacy_append_pickle(path: Path) -> List:
    if not path.exists():
        return []
    chunks = []
    with path.open("rb") as f:
        while True:
            try:
                chunks.append(pickle.load(f))
            except EOFError:
                break
    return chunks


def completed_sample_ids(path: Path) -> set:
    ids = set()
    for row in read_jsonl(path):
        if "sample_id" in row:
            ids.add(str(row["sample_id"]))
    return ids


def records_to_csv(path: Path, records: Sequence[Dict]) -> None:
    serializable = []
    for row in records:
        clean = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                clean[key] = json.dumps(value, ensure_ascii=False)
            else:
                clean[key] = value
        serializable.append(clean)
    write_csv(path, serializable)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True

