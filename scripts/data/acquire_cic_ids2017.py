"""CIC-IDS2017 dataset acquisition and verification script.

Downloads are NOT performed by this script: the official CIC-IDS2017 dataset
requires a registration form at https://cicresearch.ca/CICDataset/CIC-IDS-2017/
and has no direct programmatic URL. This script instead:

1. Locates already-downloaded CIC-IDS2017 archives/CSV files (local disk or
   Kaggle dataset attachment under /kaggle/input/).
2. Computes SHA-256 checksums for each file.
3. Verifies/records checksums in data/manifests/cic-ids2017.json.
4. Copies verified files into the dataset output directory
   (data/raw/cic-ids2017/ locally, /kaggle/working/data/raw/cic-ids2017/ on Kaggle).

Design constraints (SG-006):
- Never assume CUDA/GPU; this script is CPU-only.
- Never store credentials.
- Never download the full dataset into Git.
- Do not invent missing metadata or checksums — placeholders remain null
  until files are actually downloaded and verified.
- Kaggle compatible: source can be /kaggle/input/, output /kaggle/working/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_PATH = "data/manifests/cic-ids2017.json"

DEFAULT_LOCAL_OUTPUT = "data/raw/cic-ids2017"
DEFAULT_KAGGLE_OUTPUT = "/kaggle/working/data/raw/cic-ids2017"
DEFAULT_KAGGLE_INPUT = "/kaggle/input"

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass
class FileRecord:
    """Represents a single dataset file to acquire/verify."""

    name: str
    day: Optional[str]
    description: str
    sha256: Optional[str] = None
    sha256_verified: bool = False
    size_bytes: Optional[int] = None
    size_verified: bool = False
    record_count: Optional[int] = None
    verified: bool = False
    source_path: Optional[Path] = None
    output_path: Optional[Path] = None


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load the CIC-IDS2017 manifest JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    """Persist the manifest JSON (pretty-printed)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def find_file_in_sources(filename: str, sources: List[Path]) -> Optional[Path]:
    """Search a list of source directories for a file by name."""
    for source in sources:
        candidate = source / filename
        if candidate.is_file():
            return candidate
        # Also check one level deep (Kaggle dataset folders)
        for sub in source.iterdir():
            if sub.is_dir():
                candidate = sub / filename
                if candidate.is_file():
                    return candidate
    return None


def build_file_records(
    manifest: Dict[str, Any],
    sources: List[Path],
    output_dir: Path,
) -> List[FileRecord]:
    """Build FileRecord objects from the manifest, locating sources."""
    records: List[FileRecord] = []
    for entry in manifest.get("csv_files", []):
        name = entry["name"]
        source_path = find_file_in_sources(name, sources)
        output_path = output_dir / name
        records.append(
            FileRecord(
                name=name,
                day=entry.get("day"),
                description=entry.get("description", ""),
                sha256=entry.get("sha256"),
                sha256_verified=bool(entry.get("sha256_verified", False)),
                size_bytes=entry.get("size_bytes"),
                size_verified=bool(entry.get("size_verified", False)),
                record_count=entry.get("record_count"),
                verified=bool(entry.get("verified", False)),
                source_path=source_path,
                output_path=output_path,
            )
        )
    return records


def verify_record(record: FileRecord, output_dir: Path) -> Dict[str, Any]:
    """Verify a single file: checksum, size, record count.

    Returns a result dict with passed/failed checks and computed values.
    """
    result: Dict[str, Any] = {
        "name": record.name,
        "checksum_computed": None,
        "checksum_match": None,
        "size_bytes": None,
        "size_match": None,
        "record_count": None,
        "errors": [],
    }

    src = record.source_path
    if src is None or not src.is_file():
        result["errors"].append(f"Source file not found: {record.name}")
        return result

    # Compute checksum
    try:
        computed = sha256_of_file(src)
        result["checksum_computed"] = computed
        if record.sha256:
            result["checksum_match"] = computed == record.sha256
        else:
            result["checksum_match"] = None  # no expected checksum to compare
    except Exception as exc:  # pragma: no cover - IO dependent
        result["errors"].append(f"Checksum computation failed: {exc}")

    # Compute size
    try:
        size = src.stat().st_size
        result["size_bytes"] = size
        if record.size_bytes is not None:
            result["size_match"] = size == record.size_bytes
        else:
            result["size_match"] = None
    except Exception as exc:  # pragma: no cover - IO dependent
        result["errors"].append(f"Size computation failed: {exc}")

    # Count CSV records (header + rows)
    try:
        import csv  # local import to keep stdlib-only default path light

        with open(src, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            count = sum(1 for _ in reader)
        result["record_count"] = count
        result["header"] = header
    except Exception as exc:  # pragma: no cover - IO dependent
        result["errors"].append(f"Record count failed: {exc}")

    return result


def copy_to_output(record: FileRecord) -> Optional[str]:
    """Copy a verified source file to the output directory. Returns error or None."""
    src = record.source_path
    dst = record.output_path
    if src is None or dst is None:
        return "Missing source or output path"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        return None
    except Exception as exc:  # pragma: no cover - IO dependent
        return str(exc)


def update_manifest_from_records(
    manifest: Dict[str, Any],
    records: List[FileRecord],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Write verification results back into the manifest structure."""
    result_by_name = {r["name"]: r for r in results}

    for entry in manifest.get("csv_files", []):
        name = entry["name"]
        result = result_by_name.get(name)
        if result is None:
            continue
        entry["sha256"] = result.get("checksum_computed") or entry.get("sha256")
        entry["sha256_verified"] = bool(
            result.get("checksum_match") is not False and not result.get("errors")
        )
        entry["size_bytes"] = result.get("size_bytes") or entry.get("size_bytes")
        entry["size_verified"] = bool(
            result.get("size_match") is not False and not result.get("errors")
        )
        entry["record_count"] = result.get("record_count") or entry.get("record_count")
        entry["verified"] = bool(
            result.get("checksum_match") is not False
            and result.get("size_match") is not False
            and not result.get("errors")
        )

    manifest["verification"]["checksums_recorded"] = any(
        e.get("sha256") for e in manifest.get("csv_files", [])
    )
    return manifest


def detect_environment() -> Dict[str, Any]:
    """Detect runtime environment (Python version, Kaggle presence)."""
    import platform
    import sys

    env: Dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "is_kaggle": os.path.isdir("/kaggle"),
        "kaggle_input_exists": os.path.isdir(DEFAULT_KAGGLE_INPUT),
        "kaggle_working_exists": os.path.isdir("/kaggle/working"),
    }
    return env


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and verify CIC-IDS2017 dataset files."
    )
    parser.add_argument(
        "--manifest",
        default=MANIFEST_PATH,
        help="Path to the dataset manifest JSON.",
    )
    parser.add_argument(
        "--source",
        nargs="*",
        default=[],
        help="Source directory(s) containing downloaded dataset files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for verified files "
        "(defaults to data/raw/cic-ids2017 locally or /kaggle/working/... on Kaggle).",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Run in Kaggle mode (source=/kaggle/input, output=/kaggle/working/...).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy verified files to the output directory after verification.",
    )
    args = parser.parse_args(argv)

    env = detect_environment()

    # Resolve output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.kaggle or env["is_kaggle"]:
        output_dir = Path(DEFAULT_KAGGLE_OUTPUT)
        if not args.source:
            args.source = [DEFAULT_KAGGLE_INPUT]
    else:
        output_dir = Path(DEFAULT_LOCAL_OUTPUT)

    # Resolve source directories
    sources = [Path(s) for s in args.source] if args.source else [Path.cwd()]
    if not sources:
        sources = [Path.cwd()]

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path

    if not manifest_path.is_file():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    print("=== CIC-IDS2017 Acquisition & Verification ===")
    print(f"Environment: {env['platform']}")
    print(f"Python: {env['python_version'].split()[0]}")
    print(f"Kaggle detected: {env['is_kaggle']}")
    print(f"Sources: {[str(s) for s in sources]}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print()

    manifest = load_manifest(manifest_path)
    records = build_file_records(manifest, sources, output_dir)

    # Verify each record
    results: List[Dict[str, Any]] = []
    for record in records:
        print(f"Verifying {record.name} ...")
        if record.source_path is None:
            print(f"  SKIP: source file not found in {[str(s) for s in sources]}")
            results.append(
                {
                    "name": record.name,
                    "checksum_computed": None,
                    "checksum_match": None,
                    "size_bytes": None,
                    "size_match": None,
                    "record_count": None,
                    "errors": ["source file not found"],
                }
            )
            continue
        result = verify_record(record, output_dir)
        results.append(result)
        print(f"  checksum={result['checksum_computed']}")
        print(f"  checksum_match={result['checksum_match']}")
        print(f"  size_bytes={result['size_bytes']}")
        print(f"  size_match={result['size_match']}")
        print(f"  record_count={result['record_count']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"  ERROR: {err}")
        print()

    # Copy verified files if requested
    if args.copy:
        for record, result in zip(records, results):
            if record.source_path is None:
                continue
            if result["errors"]:
                print(f"Skipping copy of {record.name} due to verification errors.")
                continue
            err = copy_to_output(record)
            if err:
                print(f"  Copy failed for {record.name}: {err}")
            else:
                print(f"  Copied {record.name} -> {record.output_path}")

    # Update manifest and persist
    updated_manifest = update_manifest_from_records(manifest, records, results)
    save_manifest(manifest_path, updated_manifest)
    print(f"Manifest updated: {manifest_path}")

    # Summary
    total = len(records)
    verified = sum(1 for r in results if not r["errors"])
    print()
    print(f"Summary: {verified}/{total} files verified without errors.")
    if verified < total:
        print("Some files could not be verified. Re-run after placing all archives in --source.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
