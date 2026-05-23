"""Build `meta/stats.parquet` — per-column min/max/mean/std/q01/q99.

Welford streaming pass for mean+std; q01/q99 approximated by reservoir
sampling so memory stays bounded on large datasets. List columns
(joint vectors, positions, quaternions) are expanded by index, so
`state.PSM1.joint_position` produces rows
`state.PSM1.joint_position[0]` through `[5]`.
"""
from __future__ import annotations
import logging
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dvrk_data_processing.surgsync.schema import build_stats_schema


log = logging.getLogger(__name__)

# Reservoir size for q01/q99 estimation. 10k samples gives ±0.5 pct
# error on percentile estimates — fine for normalization presets.
RESERVOIR_SIZE = 10_000


class _Welford:
    """Streaming mean + variance + min/max + null count."""
    __slots__ = ("count", "null_count", "mean", "M2", "min", "max", "reservoir")

    def __init__(self):
        self.count = 0
        self.null_count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.reservoir: list[float] = []

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x
        # Reservoir sampling.
        if len(self.reservoir) < RESERVOIR_SIZE:
            self.reservoir.append(x)
        else:
            j = random.randint(0, self.count - 1)
            if j < RESERVOIR_SIZE:
                self.reservoir[j] = x

    def update_array(self, arr: np.ndarray) -> None:
        """Vectorized update on a 1-D float array — used to bulk-load a
        column rather than looping in Python."""
        finite = arr[np.isfinite(arr)]
        n_new = finite.size
        if n_new == 0:
            return
        # Update min/max cheaply.
        if finite.min() < self.min:
            self.min = float(finite.min())
        if finite.max() > self.max:
            self.max = float(finite.max())
        # Update mean/variance via combined-stats formulas.
        # Reference: Chan, Golub & LeVeque (1979) — pairwise Welford.
        new_mean = float(finite.mean())
        new_var = float(finite.var() * n_new)   # sum-of-squared-deltas
        if self.count == 0:
            self.count = n_new
            self.mean = new_mean
            self.M2 = new_var
        else:
            delta = new_mean - self.mean
            tot = self.count + n_new
            self.mean = self.mean + delta * n_new / tot
            self.M2 = self.M2 + new_var + delta**2 * self.count * n_new / tot
            self.count = tot
        # Reservoir — sample uniformly from the batch when adding to a
        # near-full reservoir. Cheap approximation: just append while
        # the reservoir has room.
        room = RESERVOIR_SIZE - len(self.reservoir)
        if room > 0:
            self.reservoir.extend(finite[:room].tolist())

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.count - 1))

    def quantile(self, q: float) -> float:
        if not self.reservoir:
            return 0.0
        return float(np.percentile(self.reservoir, q * 100.0))

    def to_row(self, column_name: str, dtype: str) -> dict[str, Any]:
        return {
            "column_name": column_name,
            "dtype":       dtype,
            "count":       self.count,
            "null_count":  self.null_count,
            "min":         self.min if self.count else None,
            "max":         self.max if self.count else None,
            "mean":        self.mean if self.count else None,
            "std":         self.std() if self.count else None,
            "q01":         self.quantile(0.01) if self.reservoir else None,
            "q99":         self.quantile(0.99) if self.reservoir else None,
            "vocab_size":  None,
        }


_PER_MODALITY_PARQUETS = (
    "timestamp.parquet", "ECM.parquet", "PSM1.parquet", "PSM2.parquet",
    "annotation.parquet",
)


def _iter_frames_parquets(dataset_root: Path) -> Iterable[Path]:
    """Yield every per-modality parquet under every episode dir.

    Each column lands in stats.parquet under its source-parquet column
    name. There's no namespace collision because the column names
    across the five parquets are disjoint (each carries its own arm /
    modality prefix).
    """
    for dataset_dir in dataset_root.iterdir():
        if not dataset_dir.is_dir() or not (dataset_dir / "episodes").is_dir():
            continue
        for task_dir in (dataset_dir / "episodes").iterdir():
            if not task_dir.is_dir():
                continue
            for ep_dir in task_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                for parquet_name in _PER_MODALITY_PARQUETS:
                    p = ep_dir / parquet_name
                    if p.is_file():
                        yield p


def _column_type_str(arr_type: pa.DataType) -> str:
    return str(arr_type)


def _is_list_type(arr_type: pa.DataType) -> bool:
    return pa.types.is_list(arr_type) or pa.types.is_fixed_size_list(arr_type)


def build_stats(dataset_root: Path) -> dict:
    """Stream every frames.parquet, compute per-column stats, write
    `meta/stats.parquet`.

    For numeric scalar columns we accumulate Welford stats directly.
    For list-of-float columns we expand by element index so
    `<col>[0]`, `<col>[1]`, ... each get their own row. String columns
    accumulate vocab_size (distinct value count) instead of numeric
    stats.
    """
    dataset_root = Path(dataset_root)

    scalar_acc: dict[str, _Welford] = {}
    scalar_dtypes: dict[str, str] = {}
    list_acc: dict[str, dict[int, _Welford]] = {}     # col → {index → Welford}
    list_dtypes: dict[str, str] = {}
    string_vocab: dict[str, set[str]] = {}
    string_null_count: dict[str, int] = {}
    string_count: dict[str, int] = {}

    n_files = 0
    for parquet_path in _iter_frames_parquets(dataset_root):
        n_files += 1
        pf = pq.ParquetFile(parquet_path)
        schema = pf.schema_arrow
        for batch in pf.iter_batches(batch_size=4096):
            for col_name in batch.schema.names:
                col = batch.column(col_name)
                col_type = col.type
                if pa.types.is_string(col_type):
                    # vocab + null counts
                    string_vocab.setdefault(col_name, set())
                    string_null_count[col_name] = string_null_count.get(col_name, 0) + col.null_count
                    string_count[col_name] = string_count.get(col_name, 0) + (len(col) - col.null_count)
                    for v in col.to_pylist():
                        if v is not None:
                            string_vocab[col_name].add(v)
                elif _is_list_type(col_type):
                    # Expand by element index. Build a list-of-lists then
                    # transpose to per-index float arrays.
                    py = col.to_pylist()
                    list_dtypes[col_name] = _column_type_str(col_type.value_type)
                    per_idx = list_acc.setdefault(col_name, {})
                    # Bulk-add rows that share the same length.
                    width: int | None = None
                    for row in py:
                        if row is None:
                            continue
                        if width is None:
                            width = len(row)
                        # Tolerate mixed widths — only process up to the
                        # length of this row.
                        for i, v in enumerate(row):
                            try:
                                vf = float(v)
                            except (TypeError, ValueError):
                                continue
                            if i not in per_idx:
                                per_idx[i] = _Welford()
                            if not math.isfinite(vf):
                                per_idx[i].null_count += 1
                            else:
                                per_idx[i].update(vf)
                elif (pa.types.is_integer(col_type) or pa.types.is_floating(col_type)
                      or pa.types.is_boolean(col_type)):
                    arr = col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
                    # Boolean columns: cast to 0/1, OK to include in stats.
                    acc = scalar_acc.setdefault(col_name, _Welford())
                    scalar_dtypes[col_name] = _column_type_str(col_type)
                    acc.null_count += col.null_count
                    acc.update_array(arr)
                # Skip other types (nested struct, bytes, etc.)

    if n_files == 0:
        log.warning("build_stats found no frames.parquet under %s", dataset_root)
        return {"n_columns": 0}

    rows: list[dict[str, Any]] = []
    for col, acc in sorted(scalar_acc.items()):
        rows.append(acc.to_row(col, scalar_dtypes[col]))
    for col, per_idx in sorted(list_acc.items()):
        for i in sorted(per_idx):
            row = per_idx[i].to_row(f"{col}[{i}]", list_dtypes[col])
            rows.append(row)
    for col, vocab in sorted(string_vocab.items()):
        rows.append({
            "column_name": col,
            "dtype":       "string",
            "count":       string_count.get(col, 0),
            "null_count":  string_null_count.get(col, 0),
            "min":         None, "max": None, "mean": None, "std": None,
            "q01":         None, "q99":  None,
            "vocab_size":  len(vocab),
        })

    schema = build_stats_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    out = dataset_root / "meta" / "stats.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", compression_level=3)
    log.info("build_stats: %d columns → %s", len(rows), out)
    return {"n_columns": len(rows), "n_files": n_files}
