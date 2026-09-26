#!/usr/bin/env python3
"""
fetch_release_digests.py
------------------------
Fetch MD5, SHA256, and BLAKE2b digests for all Grid2Op release assets
from the PyPI JSON API and write them to a structured JSON file.

Usage:
    python fetch_release_digests.py
    python fetch_release_digests.py --package Grid2Op --output digests.json
    python fetch_release_digests.py --versions 1.9.6 1.9.7 1.9.8
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"

PACKAGETYPE_LABEL = {
    "bdist_wheel": "whl",
    "sdist": "tar.gz",
}


def fetch_pypi_data(package: str) -> dict:
    url = PYPI_JSON_URL.format(package=package)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(f"ERROR: PyPI returned HTTP {exc.code} for package '{package}'")
    except urllib.error.URLError as exc:
        sys.exit(f"ERROR: Could not reach PyPI — {exc.reason}")


def build_digest_db(
    pypi_data: dict,
    only_versions: list[str] | None = None,
) -> dict:
    """
    Returns a dict structured as:
    {
        "package": "Grid2Op",
        "generated_at": "<ISO-8601 UTC>",
        "releases": {
            "1.9.8": [
                {
                    "filename": "Grid2Op-1.9.8-py3-none-any.whl",
                    "format": "whl",
                    "upload_time": "2024-01-26T10:15:04Z",
                    "yanked": false,
                    "digests": {
                        "md5":       "...",
                        "sha256":    "...",
                        "blake2b_256": "..."
                    }
                },
                ...
            ],
            ...
        }
    }
    """
    package_name = pypi_data["info"]["name"]
    releases_raw = pypi_data["releases"]

    versions = sorted(releases_raw.keys()) if only_versions is None else only_versions
    unknown = set(versions) - set(releases_raw.keys())
    if unknown:
        sys.exit(f"ERROR: Unknown version(s): {', '.join(sorted(unknown))}")

    releases_out: dict[str, list] = {}

    for version in versions:
        files = releases_raw[version]
        if not files:
            # version exists but has no files (rare edge case)
            releases_out[version] = []
            continue

        entries = []
        for f in files:
            fmt = PACKAGETYPE_LABEL.get(f.get("packagetype", ""), f.get("packagetype", "unknown"))
            entries.append(
                {
                    "filename": f["filename"],
                    "format": fmt,
                    "upload_time": f.get("upload_time_iso_8601", f.get("upload_time", "")),
                    "yanked": f.get("yanked", False),
                    "yanked_reason": f.get("yanked_reason"),
                    "digests": {
                        "md5":          f["digests"].get("md5", ""),
                        "sha256":       f["digests"].get("sha256", ""),
                        "blake2b_256":  f["digests"].get("blake2b_256", ""),
                    },
                }
            )
        releases_out[version] = entries

    return {
        "package": package_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "releases": releases_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch PyPI release digests for a package and write them to JSON."
    )
    parser.add_argument(
        "--package",
        default="Grid2Op",
        help="PyPI package name (default: Grid2Op)",
    )
    parser.add_argument(
        "--output",
        default="release_digests.json",
        help="Output JSON file path (default: release_digests.json)",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        metavar="VERSION",
        default=None,
        help="Restrict output to specific version(s). Default: all versions.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2)",
    )
    args = parser.parse_args()

    print(f"Fetching PyPI data for '{args.package}'...")
    pypi_data = fetch_pypi_data(args.package)

    print("Building digest database...")
    db = build_digest_db(pypi_data, only_versions=args.versions)

    n_versions = len(db["releases"])
    n_files = sum(len(v) for v in db["releases"].values())
    print(f"  → {n_versions} version(s), {n_files} file(s)")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(db, indent=args.indent), encoding="utf-8")
    print(f"Written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
