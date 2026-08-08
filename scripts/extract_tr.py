#!/usr/bin/env python3
"""
Extract each subject's repetition time (TR) from its func_preproc NIfTI header.

Reads only the 348-byte NIfTI-1 header (via gzip streaming, no full
decompression needed) for every subject in the phenotypic table, and writes
one row per subject to data/raw/subject_tr.csv.

Reuses downloadData.py's phenotypic-CSV fetch and no_filename exclusion
check, so the subject list here is identical to the one used for downloading
rois_aal/func_preproc/func_mask.

Usage:
    python scripts/extract_tr.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "download"))

from downloadData import download_phenotypic_csv, load_subjects  # noqa: E402

# NIfTI-1 header field offsets (bytes)
DIM_OFFSET = 40           # short[8]
PIXDIM_OFFSET = 76        # float[8]
XYZT_UNITS_OFFSET = 123   # unsigned char
HEADER_SIZE = 348

TIME_UNITS_SEC = 8
TIME_UNITS_MSEC = 16
TIME_UNITS_USEC = 24


def read_tr_and_volumes(nifti_gz_path: Path) -> tuple[float, int]:
    """Read TR (seconds) and volume count from a gzipped NIfTI-1 header."""

    with gzip.open(nifti_gz_path, "rb") as f:
        header = f.read(HEADER_SIZE)

    dim = struct.unpack_from("<8h", header, DIM_OFFSET)
    pixdim = struct.unpack_from("<8f", header, PIXDIM_OFFSET)
    xyzt_units = struct.unpack_from("<B", header, XYZT_UNITS_OFFSET)[0]

    n_dims = dim[0]
    n_volumes = dim[4] if n_dims >= 4 else 1
    tr_raw = pixdim[4]

    time_units = xyzt_units & 0x38

    if time_units == TIME_UNITS_MSEC:
        tr_seconds = tr_raw / 1000.0
    elif time_units == TIME_UNITS_USEC:
        tr_seconds = tr_raw / 1_000_000.0
    else:
        # TIME_UNITS_SEC or unspecified (0): C-PAC fMRI headers store TR in
        # seconds even when the units flag isn't set.
        tr_seconds = tr_raw

    return tr_seconds, n_volumes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root data directory containing func_preproc/. Default: data/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <data-dir>/subject_tr.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = args.data_dir.resolve()
    output_path = args.output or (data_dir / "subject_tr.csv")
    func_preproc_dir = data_dir / "func_preproc"

    phenotypic_csv = download_phenotypic_csv(
        data_dir / "metadata", retries=5, timeout=180, overwrite=False
    )
    subjects = load_subjects(phenotypic_csv, requested_sites=None, max_subjects=None)
    logging.info("Subjects to process: %d", len(subjects))

    rows = []
    missing = 0

    for i, subject in enumerate(subjects, start=1):
        nifti_path = func_preproc_dir / f"{subject.file_id}_func_preproc.nii.gz"

        if not nifti_path.exists():
            logging.warning("[%d/%d] %s: file not found, skipping", i, len(subjects), subject.file_id)
            missing += 1
            continue

        tr_seconds, n_volumes = read_tr_and_volumes(nifti_path)

        rows.append(
            {
                "FILE_ID": subject.file_id,
                "SITE_ID": subject.site_id,
                "DX_GROUP": subject.dx_group,
                "TR_seconds": tr_seconds,
                "N_VOLUMES": n_volumes,
            }
        )

        if i % 100 == 0:
            logging.info("[%d/%d] processed", i, len(subjects))

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FILE_ID", "SITE_ID", "DX_GROUP", "TR_seconds", "N_VOLUMES"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Wrote %d records to %s", len(rows), output_path)

    if missing:
        logging.warning("%d subjects had no func_preproc file and were skipped.", missing)

    if len(rows) == len(subjects):
        logging.info("VERIFY PASS: %d records match %d expected subjects.", len(rows), len(subjects))
        return 0

    logging.error(
        "VERIFY FAIL: %d records written, expected %d subjects.", len(rows), len(subjects)
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
