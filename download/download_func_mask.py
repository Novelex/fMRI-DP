#!/usr/bin/env python3
"""
Download the ABIDE-I PCP func_mask derivative (C-PAC, nofilt_noglobal).

Needed for the ALFF/mALFF pipeline (DPARSFA requires a brain mask on the
same grid as func_preproc). Reuses downloadData.py's phenotypic-CSV fetch,
no_filename exclusion check, and resumable download logic so the subject
list stays identical to the rois_aal/func_preproc download.

Usage:
    python download/download_func_mask.py
    python download/download_func_mask.py --max-subjects 5 --workers 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from downloadData import (
    PIPELINE,
    STRATEGY,
    build_tasks,
    download_phenotypic_csv,
    load_subjects,
    download_streaming,
    write_download_manifest,
)

DERIVATIVE = "func_mask"
EXTENSION = ".nii.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory. Default: data/raw (same root as downloadData.py)",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Optional subset of sites to restrict to, e.g. --sites NYU PITT.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit the number of participants for testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of simultaneous downloads. Default: 3",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum download attempts per file. Default: 5",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Network timeout in seconds. Default: 180",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Output directory: %s", output_dir)
    logging.info("Pipeline: %s | Strategy: %s | Derivative: %s", PIPELINE, STRATEGY, DERIVATIVE)

    phenotypic_csv = download_phenotypic_csv(
        metadata_dir,
        retries=args.retries,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )

    subjects = load_subjects(
        phenotypic_csv,
        requested_sites=args.sites,
        max_subjects=args.max_subjects,
    )
    expected_count = len(subjects)
    logging.info("Participants selected (post no_filename check): %d", expected_count)

    tasks = build_tasks(subjects, output_dir, {DERIVATIVE: EXTENSION})
    logging.info("Derivative files requested: %d", len(tasks))

    manifest_rows: list[dict[str, str]] = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                download_streaming,
                task.url,
                task.destination,
                retries=args.retries,
                timeout=args.timeout,
                overwrite=args.overwrite,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1

            try:
                status, size_bytes = future.result()
                error = ""
                logging.info(
                    "[%d/%d] %s | %s | %s | %.2f MiB",
                    completed, len(tasks), task.file_id, task.derivative, status,
                    size_bytes / (1024 ** 2),
                )
            except Exception as exc:
                status = "failed"
                size_bytes = 0
                error = str(exc)
                failed += 1
                logging.error(
                    "[%d/%d] %s | %s | FAILED | %s",
                    completed, len(tasks), task.file_id, task.derivative, exc,
                )

            manifest_rows.append(
                {
                    "FILE_ID": task.file_id,
                    "SITE_ID": task.site_id,
                    "DX_GROUP": task.dx_group,
                    "pipeline": PIPELINE,
                    "strategy": STRATEGY,
                    "derivative": task.derivative,
                    "local_path": str(task.destination),
                    "url": task.url,
                    "status": status,
                    "size_bytes": str(size_bytes),
                    "error": error,
                }
            )

    manifest_rows.sort(key=lambda row: row["FILE_ID"])
    manifest_path = output_dir / f"download_manifest_{DERIVATIVE}.csv"
    write_download_manifest(manifest_rows, manifest_path)

    successful = len(tasks) - failed
    logging.info("Download finished: %d successful, %d failed.", successful, failed)
    logging.info("Manifest: %s", manifest_path)

    actual_count = sum(
        1 for p in (output_dir / DERIVATIVE).glob(f"*_{DERIVATIVE}{EXTENSION}")
        if p.stat().st_size > 0
    )

    if actual_count == expected_count:
        logging.info(
            "VERIFY PASS: %d %s files on disk match %d expected subjects.",
            actual_count, DERIVATIVE, expected_count,
        )
    else:
        logging.error(
            "VERIFY FAIL: %d %s files on disk, expected %d subjects.",
            actual_count, DERIVATIVE, expected_count,
        )
        return 1

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        raise SystemExit(1)
