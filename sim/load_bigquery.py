"""
Load the generated feed into BigQuery.

The local DuckDB path is the default for a reason -- it needs no account and it is
what CI runs -- but the dbt models are written in cross-dialect SQL and the
BigQuery target is meant to actually work, so this is the loader that makes it
work rather than a line in the README promising one.

    python -m sim.load_bigquery --dataset jaffle_sim_raw
    python -m sim.load_bigquery --dataset jaffle_sim_raw --project my-gcp-project

Then point dbt at wherever the tables landed:

    dbt build --target bigquery \\
      --vars '{raw_database: my-gcp-project, raw_schema: jaffle_sim_raw}'

Requires `pip install google-cloud-bigquery` and GOOGLE_APPLICATION_CREDENTIALS
(or DBT_GCP_KEYFILE) pointing at a service-account key. Neither is in
requirements.txt: this is an optional path, and a reviewer running the project
locally should not have to install a cloud SDK to do it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="Load the generated parquet feed into BigQuery.")
    ap.add_argument("--dataset", required=True, help="target BigQuery dataset, e.g. jaffle_sim_raw")
    ap.add_argument(
        "--project",
        default=os.environ.get("DBT_GCP_PROJECT"),
        help="GCP project id (defaults to $DBT_GCP_PROJECT)",
    )
    ap.add_argument("--location", default=os.environ.get("DBT_GCP_LOCATION", "US"))
    ap.add_argument(
        "--replace",
        action="store_true",
        help="truncate each table before loading instead of appending",
    )
    args = ap.parse_args()

    if not args.project:
        print("error: --project not given and DBT_GCP_PROJECT is not set", file=sys.stderr)
        return 2

    try:
        from google.cloud import bigquery
    except ImportError:
        print(
            "error: google-cloud-bigquery is not installed.\n"
            "       pip install google-cloud-bigquery\n"
            "       (kept out of requirements.txt so the local DuckDB path stays "
            "dependency-light)",
            file=sys.stderr,
        )
        return 2

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    raw_dir = ROOT / cfg["output"]["raw_dir"]
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        print(
            f"error: no parquet files in {raw_dir}. Run `python -m sim.generate` first.",
            file=sys.stderr,
        )
        return 1

    client = bigquery.Client(project=args.project)
    dataset_ref = bigquery.Dataset(f"{args.project}.{args.dataset}")
    dataset_ref.location = args.location
    client.create_dataset(dataset_ref, exists_ok=True)
    print(f"dataset {args.project}.{args.dataset} ready ({args.location})")

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        # Parquet carries its own schema, so autodetect is not a guess here --
        # the column types come from the file, not from sampling rows.
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if args.replace
            else bigquery.WriteDisposition.WRITE_EMPTY
        ),
    )

    for path in files:
        table_id = f"{args.project}.{args.dataset}.{path.stem}"
        with path.open("rb") as fh:
            job = client.load_table_from_file(fh, table_id, job_config=job_config)
        job.result()
        table = client.get_table(table_id)
        print(f"  loaded {path.stem:<26} {table.num_rows:>10,} rows")

    print(
        "\nNow run:\n"
        f"  dbt build --target bigquery \\\n"
        f"    --vars '{{raw_database: {args.project}, raw_schema: {args.dataset}}}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
