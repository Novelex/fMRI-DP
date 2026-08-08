#!/usr/bin/env python3
"""
Download ABIDE-I PCP derivatives from the public FCP-INDI S3 bucket.

Default configuration
---------------------
Pipeline:
    C-PAC

Strategy:
    nofilt_noglobal

Downloaded derivatives:
    rois_aal      (.1D)
    func_preproc  (.nii.gz)

Default output:
    data/raw/

Examples
--------
Test with five participants:

    python download/downloadData.py \
        --max-subjects 5 \
        --workers 2

Download the full cohort (every site):

    python download/downloadData.py \
        --workers 4

Download selected sites:

    python download/downloadData.py \
        --sites NYU PITT \
        --workers 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------
# PCP configuration
# ---------------------------------------------------------------------

S3_ROOT = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative"
)

PHENOTYPIC_URL = (
    f"{S3_ROOT}/Phenotypic_V1_0b_preprocessed1.csv"
)

PIPELINE = "cpac"
STRATEGY = "nofilt_noglobal"

VALID_STRATEGIES = [
    "nofilt_noglobal",
    "nofilt_global",
    "filt_noglobal",
    "filt_global",
]

DEFAULT_DERIVATIVES = {
    "rois_aal": ".1D",
    "func_preproc": ".nii.gz",
}


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Subject:
    file_id: str
    site_id: str
    dx_group: str


@dataclass(frozen=True)
class DownloadTask:
    file_id: str
    site_id: str
    dx_group: str
    derivative: str
    url: str
    destination: Path


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ABIDE-I PCP C-PAC nofilt_noglobal derivatives: "
            "rois_aal and func_preproc."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory. Default: data/raw",
    )

    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help=(
            "Optional subset of sites to restrict to, for example "
            "--sites NYU PITT. Default: every site in the phenotypic table."
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=VALID_STRATEGIES,
        default=STRATEGY,
        help=f"PCP denoising strategy. Default: {STRATEGY}",
    )

    parser.add_argument(
        "--derivatives",
        nargs="+",
        choices=list(DEFAULT_DERIVATIVES.keys()),
        default=list(DEFAULT_DERIVATIVES.keys()),
        help="Which derivatives to fetch. Default: both rois_aal and func_preproc.",
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
        help=(
            "Number of simultaneous downloads. "
            "Default: 3"
        ),
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


# ---------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------

def download_streaming(
    url: str,
    destination: Path,
    *,
    retries: int,
    timeout: int,
    overwrite: bool,
) -> tuple[str, int]:
    """
    Download one file using a temporary .part file.

    Existing completed files are skipped unless --overwrite is used.

    Returns
    -------
    status:
        downloaded, resumed or skipped

    size_bytes:
        Final local file size
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    if (
        destination.exists()
        and destination.stat().st_size > 0
        and not overwrite
    ):
        return "skipped", destination.stat().st_size

    part_path = destination.with_name(
        destination.name + ".part"
    )

    if overwrite:
        destination.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        resume_from = (
            part_path.stat().st_size
            if part_path.exists()
            else 0
        )

        headers = {
            "User-Agent": "ABIDE-PCP-downloader/2.0"
        }

        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=timeout) as response:
                status_code = getattr(response, "status", 200)

                # If the server ignores the Range request,
                # restart instead of appending duplicate data.
                if resume_from > 0 and status_code != 206:
                    write_mode = "wb"
                    resume_from = 0
                else:
                    write_mode = "ab" if resume_from > 0 else "wb"

                with part_path.open(write_mode) as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        output_file.write(chunk)

            if (
                not part_path.exists()
                or part_path.stat().st_size == 0
            ):
                raise OSError("Downloaded file is empty.")

            os.replace(part_path, destination)

            status = (
                "resumed"
                if resume_from > 0
                else "downloaded"
            )

            return status, destination.stat().st_size

        except HTTPError as exc:
            if exc.code == 416:
                logging.warning(
                    "HTTP 416 for %s. Removing stale partial file.",
                    destination.name,
                )
                part_path.unlink(missing_ok=True)

            elif exc.code in {403, 404}:
                raise RuntimeError(
                    f"Remote file unavailable "
                    f"(HTTP {exc.code}): {url}"
                ) from exc

            else:
                logging.warning(
                    "Attempt %d/%d failed for %s: HTTP %s",
                    attempt,
                    retries,
                    destination.name,
                    exc.code,
                )

        except (
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            logging.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                retries,
                destination.name,
                exc,
            )

        if attempt < retries:
            sleep_seconds = min(2 ** (attempt - 1), 30)
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Failed after {retries} attempts: {url}"
    )


def download_phenotypic_csv(
    metadata_dir: Path,
    *,
    retries: int,
    timeout: int,
    overwrite: bool,
) -> Path:
    """Download the ABIDE-I preprocessed phenotypic table."""

    csv_path = (
        metadata_dir
        / "Phenotypic_V1_0b_preprocessed1.csv"
    )

    status, size_bytes = download_streaming(
        PHENOTYPIC_URL,
        csv_path,
        retries=retries,
        timeout=timeout,
        overwrite=overwrite,
    )

    logging.info(
        "Phenotypic CSV: %s | %s | %.2f MiB",
        csv_path,
        status,
        size_bytes / (1024 ** 2),
    )

    return csv_path


# ---------------------------------------------------------------------
# Participant selection
# ---------------------------------------------------------------------

def get_site_filter(
    requested_sites: Iterable[str] | None,
) -> set[str] | None:
    """
    Determine which sites should be downloaded.

    Default:
        No filter — every site in the phenotypic table.

    --sites:
        Restrict to only the requested sites.
    """

    if requested_sites:
        return {
            site.strip().upper()
            for site in requested_sites
        }

    return None


def load_subjects(
    csv_path: Path,
    *,
    requested_sites: Iterable[str] | None,
    max_subjects: int | None,
) -> list[Subject]:
    """Load and filter participants from the phenotypic CSV."""

    site_filter = get_site_filter(requested_sites)

    subjects: list[Subject] = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "FILE_ID",
            "SITE_ID",
            "DX_GROUP",
        }

        available_columns = set(reader.fieldnames or [])

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Required columns missing from phenotypic CSV: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            file_id = (
                row.get("FILE_ID") or ""
            ).strip()

            site_id = (
                row.get("SITE_ID") or ""
            ).strip()

            dx_group = (
                row.get("DX_GROUP") or ""
            ).strip()

            # Exclusion criterion:
            # FILE_ID must not equal no_filename.
            if (
                not file_id
                or file_id.lower() == "no_filename"
            ):
                continue

            site_upper = site_id.upper()

            if (
                site_filter is not None
                and site_upper not in site_filter
            ):
                continue

            subjects.append(
                Subject(
                    file_id=file_id,
                    site_id=site_id,
                    dx_group=dx_group,
                )
            )

            if (
                max_subjects is not None
                and len(subjects) >= max_subjects
            ):
                break

    if not subjects:
        raise RuntimeError(
            "No participants matched the selected criteria."
        )

    return subjects


# ---------------------------------------------------------------------
# Derivative task construction
# ---------------------------------------------------------------------

def derivative_url(
    file_id: str,
    derivative: str,
    extension: str,
) -> str:
    """Construct the PCP S3 download URL."""

    return (
        f"{S3_ROOT}/Outputs/"
        f"{PIPELINE}/"
        f"{STRATEGY}/"
        f"{derivative}/"
        f"{file_id}_{derivative}{extension}"
    )


def build_tasks(
    subjects: list[Subject],
    output_dir: Path,
    derivatives: dict[str, str],
) -> list[DownloadTask]:
    """Create one download task for every derivative and subject."""

    tasks: list[DownloadTask] = []

    for subject in subjects:
        for derivative, extension in derivatives.items():
            filename = (
                f"{subject.file_id}_"
                f"{derivative}"
                f"{extension}"
            )

            destination = (
                output_dir
                / derivative
                / filename
            )

            tasks.append(
                DownloadTask(
                    file_id=subject.file_id,
                    site_id=subject.site_id,
                    dx_group=subject.dx_group,
                    derivative=derivative,
                    url=derivative_url(
                        subject.file_id,
                        derivative,
                        extension,
                    ),
                    destination=destination,
                )
            )

    return tasks


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def write_download_manifest(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Write the download results to a CSV manifest."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "FILE_ID",
        "SITE_ID",
        "DX_GROUP",
        "pipeline",
        "strategy",
        "derivative",
        "local_path",
        "url",
        "status",
        "size_bytes",
        "error",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main() -> int:
    global STRATEGY

    args = parse_args()
    STRATEGY = args.strategy

    if args.workers < 1:
        raise ValueError(
            "--workers must be at least 1."
        )

    if args.retries < 1:
        raise ValueError(
            "--retries must be at least 1."
        )

    if (
        args.max_subjects is not None
        and args.max_subjects < 1
    ):
        raise ValueError(
            "--max-subjects must be at least 1."
        )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info(
        "Output directory: %s",
        output_dir,
    )

    logging.info(
        "Pipeline: %s | Strategy: %s",
        PIPELINE,
        STRATEGY,
    )

    derivatives = {d: DEFAULT_DERIVATIVES[d] for d in args.derivatives}

    logging.info(
        "Derivatives: %s",
        ", ".join(derivatives.keys()),
    )

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

    logging.info(
        "Participants selected: %d",
        len(subjects),
    )

    selected_sites = sorted(
        {subject.site_id for subject in subjects}
    )

    logging.info(
        "Selected sites: %s",
        ", ".join(selected_sites),
    )

    tasks = build_tasks(
        subjects,
        output_dir,
        derivatives,
    )

    logging.info(
        "Derivative files requested: %d",
        len(tasks),
    )

    manifest_rows: list[dict[str, str]] = []

    completed = 0
    failed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
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
                    completed,
                    len(tasks),
                    task.file_id,
                    task.derivative,
                    status,
                    size_bytes / (1024 ** 2),
                )

            except Exception as exc:
                status = "failed"
                size_bytes = 0
                error = str(exc)
                failed += 1

                logging.error(
                    "[%d/%d] %s | %s | FAILED | %s",
                    completed,
                    len(tasks),
                    task.file_id,
                    task.derivative,
                    exc,
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

    manifest_rows.sort(
        key=lambda row: (
            row["FILE_ID"],
            row["derivative"],
        )
    )

    manifest_suffix = "-".join(sorted(derivatives)) + f"_{STRATEGY}"
    manifest_path = (
        output_dir
        / f"download_manifest_{manifest_suffix}.csv"
    )

    write_download_manifest(
        manifest_rows,
        manifest_path,
    )

    successful = len(tasks) - failed

    logging.info(
        "Download finished: %d successful, %d failed.",
        successful,
        failed,
    )

    logging.info(
        "Manifest: %s",
        manifest_path,
    )

    for derivative in derivatives:
        logging.info(
            "%s directory: %s",
            derivative,
            output_dir / derivative,
        )

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "\nDownload interrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exc:
        logging.exception(
            "Fatal error: %s",
            exc,
        )
        raise SystemExit(1)
