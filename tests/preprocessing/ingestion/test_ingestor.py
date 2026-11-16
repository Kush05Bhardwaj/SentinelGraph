"""Unit tests for the SentinelGraph CSV ingestion module.

Covers:
    * valid CSV ingestion
    * missing files
    * empty files
    * header-only CSVs
    * invalid schemas
    * schema type casting
    * malformed rows (skip / warn / error)
    * multiple CSV files
    * encoding fallback
    * configurable input directory
    * raw source files remain unmodified
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from ml.preprocessing.ingestion import (
    CSVIngestor,
    FileStats,
    IngestionEmptyFileError,
    IngestionEncodingError,
    IngestionError,
    IngestionFileNotFoundError,
    IngestionSchemaError,
    Schema,
    ingest_csv,
    ingest_directory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.write_text(content, encoding=encoding)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


VALID_CSV = """src_ip,dst_ip,bytes_sent,label
10.0.0.1,10.0.0.2,1024,benign
10.0.0.3,10.0.0.4,2048,malicious
10.0.0.5,10.0.0.6,512,benign
"""

SECOND_CSV = """src_ip,dst_ip,bytes_sent,label
10.0.0.7,10.0.0.8,4096,malicious
10.0.0.9,10.0.0.10,8192,benign
"""


# ---------------------------------------------------------------------------
# Valid CSV
# ---------------------------------------------------------------------------

def test_valid_csv_returns_standardized_dataframe(tmp_path: Path) -> None:
    p = tmp_path / "valid.csv"
    write_csv(p, VALID_CSV)

    before_sha = file_sha256(p)
    result = ingest_csv(p)

    assert isinstance(result.dataframe, pd.DataFrame)
    assert len(result.dataframe) == 3
    assert list(result.dataframe.columns) == ["src_ip", "dst_ip", "bytes_sent", "label"]
    assert result.stats.rows_read == 3
    assert result.stats.rows_valid == 3
    assert result.stats.rows_malformed == 0
    assert result.valid is True
    assert result.stats.file_name == "valid.csv"
    assert result.stats.file_size_bytes == p.stat().st_size
    assert result.stats.encoding_used == "utf-8"

    # Raw source file must not be modified.
    assert file_sha256(p) == before_sha


def test_header_only_csv(tmp_path: Path) -> None:
    p = tmp_path / "header_only.csv"
    write_csv(p, "src_ip,dst_ip,label\n")

    result = ingest_csv(p)
    assert len(result.dataframe) == 0
    assert result.stats.rows_read == 0
    assert result.stats.rows_valid == 0
    assert result.valid is True


# ---------------------------------------------------------------------------
# Missing / empty files
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(IngestionFileNotFoundError):
        ingest_csv(missing)


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    with pytest.raises(IngestionEmptyFileError):
        ingest_csv(p)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_invalid_schema_missing_required_column(tmp_path: Path) -> None:
    p = tmp_path / "bad_schema.csv"
    write_csv(p, "src_ip,dst_ip,label\n10.0.0.1,10.0.0.2,benign\n")

    schema = Schema(required_columns=["src_ip", "dst_ip", "bytes_sent", "label"])
    with pytest.raises(IngestionSchemaError) as excinfo:
        ingest_csv(p, schema=schema)
    assert "bytes_sent" in str(excinfo.value)


def test_schema_type_casting(tmp_path: Path) -> None:
    p = tmp_path / "types.csv"
    write_csv(p, "src_ip,dst_ip,bytes_sent\n10.0.0.1,10.0.0.2,1024\n")

    schema = Schema(
        required_columns=["src_ip", "dst_ip", "bytes_sent"],
        column_types={"bytes_sent": "int64"},
    )
    result = ingest_csv(p, schema=schema)
    assert result.dataframe["bytes_sent"].dtype == "int64"
    assert result.valid is True


# ---------------------------------------------------------------------------
# Malformed rows
# ---------------------------------------------------------------------------

def test_malformed_rows_skipped(tmp_path: Path) -> None:
    # Header has 3 columns; second data row has 4 -> malformed.
    content = "src_ip,dst_ip,label\n10.0.0.1,10.0.0.2,benign\n10.0.0.3,10.0.0.4,extra,malicious\n"
    p = tmp_path / "malformed.csv"
    write_csv(p, content)

    result = ingest_csv(p, on_malformed="skip")
    assert len(result.dataframe) == 1
    assert result.stats.rows_malformed >= 1
    assert result.valid is False  # malformed rows present


def test_malformed_rows_warns(tmp_path: Path, caplog) -> None:
    content = "src_ip,dst_ip,label\n10.0.0.1,10.0.0.2,benign\n10.0.0.3,10.0.0.4,extra,malicious\n"
    p = tmp_path / "malformed_warn.csv"
    write_csv(p, content)

    caplog.set_level("WARNING")
    result = ingest_csv(p, on_malformed="warn")
    assert len(result.dataframe) == 1
    assert "malformed" in caplog.text.lower()


def test_malformed_rows_error_raises(tmp_path: Path) -> None:
    content = "src_ip,dst_ip,label\n10.0.0.1,10.0.0.2,benign\n10.0.0.3,10.0.0.4,extra,malicious\n"
    p = tmp_path / "malformed_error.csv"
    write_csv(p, content)

    with pytest.raises(IngestionSchemaError):
        ingest_csv(p, on_malformed="error")


# ---------------------------------------------------------------------------
# Encoding fallback
# ---------------------------------------------------------------------------

def test_encoding_fallback_latin1(tmp_path: Path) -> None:
    # A byte sequence valid in latin-1 (0xe9) but not utf-8 on its own.
    content = "src_ip,label\n10.0.0.1,b\u00e9nin\n"
    p = tmp_path / "latin1.csv"
    write_csv(p, content, encoding="latin-1")

    # utf-8 fails; latin-1 fallback succeeds.
    result = ingest_csv(p, encoding="utf-8", fallback_encodings=("latin-1",))
    assert len(result.dataframe) == 1
    assert result.stats.encoding_used == "latin-1"


def test_no_encoding_works_raises(tmp_path: Path) -> None:
    # latin-1 and cp1252 decode every byte, so use only utf-8-family
    # fallbacks that reject 0xff / missing BOM.
    p = tmp_path / "weird.csv"
    p.write_bytes(b"\xff\xfe\x00\x00garbage")
    with pytest.raises(IngestionEncodingError):
        ingest_csv(p, encoding="utf-8", fallback_encodings=("utf-8-sig",))


# ---------------------------------------------------------------------------
# Multiple files / configurable directory
# ---------------------------------------------------------------------------

def test_multiple_csv_files(tmp_path: Path) -> None:
    write_csv(tmp_path / "a.csv", VALID_CSV)
    write_csv(tmp_path / "b.csv", SECOND_CSV)

    ingestor = CSVIngestor(input_dir=str(tmp_path))
    results, combined = ingestor.ingest_directory()

    assert len(results) == 2
    assert {r.file_name for r in results} == {"a.csv", "b.csv"}
    assert combined is not None
    assert len(combined) == 5
    assert all(r.valid for r in results)


def test_input_dir_from_constructor(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    write_csv(p, VALID_CSV)

    ingestor = CSVIngestor(input_dir=str(tmp_path))
    result = ingestor.ingest_file(str(p))
    assert result.file_name == "x.csv"
    assert len(result.dataframe) == 3


def test_no_input_dir_raises(tmp_path: Path) -> None:
    ingestor = CSVIngestor()  # no input_dir
    with pytest.raises(IngestionError):
        ingestor.ingest_directory()


def test_directory_not_found_raises() -> None:
    ingestor = CSVIngestor(input_dir="/nonexistent/dir/for/testing")
    with pytest.raises(IngestionFileNotFoundError):
        ingestor.ingest_directory()


def test_recursive_glob(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    write_csv(tmp_path / "root.csv", VALID_CSV)
    write_csv(sub / "nested.csv", SECOND_CSV)

    ingestor = CSVIngestor(input_dir=str(tmp_path))
    results, combined = ingestor.ingest_directory(recursive=True)

    assert len(results) == 2
    assert len(combined) == 5


def test_file_stats_dataclass() -> None:
    stats = FileStats(
        file_path="/tmp/x.csv",
        file_name="x.csv",
        file_size_bytes=100,
        encoding_used="utf-8",
        rows_read=5,
        rows_valid=5,
        rows_malformed=0,
        column_count=3,
        column_names=["a", "b", "c"],
    )
    assert stats.valid is True
    d = stats.to_dict()
    assert d["file_name"] == "x.csv"
    assert d["rows_read"] == 5


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def test_ingest_directory_convenience(tmp_path: Path) -> None:
    write_csv(tmp_path / "m.csv", VALID_CSV)
    results, combined = ingest_directory(str(tmp_path))
    assert len(results) == 1
    assert combined is not None
    assert len(combined) == 3
