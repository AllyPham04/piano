"""Inspect the B-K/MAESTRO-test Parquet shards without downloading the audio (M7).

Reads only Parquet footers (schema, row groups) and one small column through HTTP
range requests.

Usage (from the repository root): python scripts/inspect_hf_mirror.py
"""

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
files = sorted(fs.glob("datasets/B-K/MAESTRO-test/data/*.parquet"))
print(len(files), "shards")
total_rows = 0
for i, f in enumerate(files):
    with fs.open(f, "rb", block_size=1 << 20) as fh:
        meta = pq.ParquetFile(fh).metadata
        sizes = [meta.row_group(g).num_rows for g in range(meta.num_row_groups)]
        total_rows += meta.num_rows
        if i == 0:
            print(pq.ParquetFile(fh).schema_arrow)
        print(f"{f.split('/')[-1]}: rows={meta.num_rows} row_groups={meta.num_row_groups} "
              f"rows/group={sizes[:6]}{'...' if len(sizes) > 6 else ''} "
              f"size={fs.info(f)['size'] / 1e6:.0f} MB")
print("total rows:", total_rows)
