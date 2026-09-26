"""
grid2op/check_release.py
------------------------
CLI subcommand for verifying the integrity of a local Grid2Op release asset
against known-good digests published on PyPI.

Invocation (once wired into __main__.py):
    grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl
    grid2op.check_release Grid2Op-1.9.8.tar.gz --sha256
    grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl --md5 --sha256 --blake2b
    grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl --all
    grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl --offline digests.json

Exit codes:
    0  all requested checks passed
    1  one or more checks failed or file not recognised
    2  usage / argument error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from grid2op.Download.DownloadDataset import ALLOWED_SCHEMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYPI_VERSION_URL = "https://pypi.org/pypi/{package}/{version}/json"
PACKAGE_NAME = "Grid2Op"


# Regex to extract version from a Grid2Op filename, e.g.:
#   Grid2Op-1.9.8-py3-none-any.whl        →  1.9.8
#   Grid2Op-1.9.8.tar.gz                  →  1.9.8
#   grid2op-1.10.5.post1-py3-none-any.whl →  1.10.5.post1
#   grid2op-1.12.2.dev0.tar.gz            →  1.12.2.dev0
#   Grid2Op-1.9.8-py3-none-any.whl        →  1.9.8
#   Grid2Op-1.9.8.tar.gz                  →  1.9.8
#   Grid2Op-1.7.2rc2.tar.gz               →  1.7.2rc2

_VERSION_RE = re.compile(
    r"^grid2op-(\d+(?:\.\d+)+(?:\.(?:post|dev)\d+|[ab]\d+|rc\d+)?)[-.]",
    re.IGNORECASE,
)

DIGEST_ALGORITHMS = {
    "md5":        ("md5",       hashlib.md5),
    "sha256":     ("sha256",    hashlib.sha256),
    "blake2b_256":("blake2b_256", lambda: hashlib.blake2b(digest_size=32)),
}

CHUNK = 1 << 20  # 1 MiB read chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_digests(path: Path, algorithms: list[str]) -> dict[str, str]:  # pragma: no cover
    hashers = {alg: DIGEST_ALGORITHMS[alg][1]() for alg in algorithms}
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            for h in hashers.values():
                h.update(chunk)
    return {alg: hashers[alg].hexdigest() for alg in algorithms}


def _fetch_pypi_digests(version: str, filename: str) -> dict[str, str] | None:  # pragma: no cover
    """Return the digest dict for *filename* from the PyPI JSON API, or None."""
    url = PYPI_VERSION_URL.format(package=PACKAGE_NAME, version=version)
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsafe URL scheme '{scheme}'. Only {ALLOWED_SCHEMES} are allowed."
        )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    for f in data.get("urls", []):
        if f["filename"] == filename:
            return f["digests"]  # keys: md5, sha256, blake2b_256
    return None


def _load_offline_digests(json_path: Path, filename: str) -> dict[str, str] | None:  # pragma: no cover
    """Return digest dict from a local JSON file produced by fetch_release_digests.py."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for _version, entries in data.get("releases", {}).items():
        for entry in entries:
            if entry["filename"] == filename:
                return entry["digests"]
    return None


def _colour(text: str, ok: bool) -> str:  # pragma: no cover
    """ANSI green for pass, red for fail — degrades gracefully on Windows."""
    if sys.stdout.isatty():
        code = "\033[32m" if ok else "\033[31m"
        return f"{code}{text}\033[0m"
    return text


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify(
    asset: Path,
    algorithms: list[str],
    offline_db: Path | None = None,
    quiet: bool = False,
) -> bool:  # pragma: no cover
    """
    Verify *asset* against PyPI (or an offline digest DB).
    Returns True if every requested check passes.
    """
    filename = asset.name

    # Extract version from filename
    m = _VERSION_RE.match(filename)
    if not m:
        print(
            f"ERROR: Cannot determine Grid2Op version from filename '{filename}'.\n"
            "Expected a name like 'Grid2Op-1.9.8-py3-none-any.whl' or 'Grid2Op-1.9.8.tar.gz'.",
            file=sys.stderr,
        )
        return False

    version = m.group(1)

    if not quiet:
        print(f"File    : {filename}")
        print(f"Version : {version}")

    # Fetch expected digests
    if offline_db is not None:
        if not quiet:
            print(f"Source  : offline ({offline_db})")
        expected = _load_offline_digests(offline_db, filename)
    else:
        if not quiet:
            print("Source  : PyPI (live)")
        try:
            expected = _fetch_pypi_digests(version, filename)
        except urllib.error.URLError as exc:
            print(
                f"ERROR: Could not reach PyPI — {exc.reason}\n"
                "Tip: use --offline <digests.json> to check without internet access.",
                file=sys.stderr,
            )
            return False

    if expected is None:
        print(
            f"ERROR: '{filename}' was not found on PyPI for version {version}.\n"
            "Double-check the filename or use --offline if this is a private build.",
            file=sys.stderr,
        )
        return False

    # Compute local digests
    if not quiet:
        print(f"Checking: {', '.join(algorithms)}\n")

    local = _compute_digests(asset, algorithms)

    all_ok = True
    for alg in algorithms:
        expected_val = expected.get(alg, "")
        local_val = local[alg]
        ok = expected_val and (local_val == expected_val)
        all_ok = all_ok and ok

        label = alg.upper().replace("_", "-")
        status = _colour("PASS", ok) if ok else _colour("FAIL", ok)

        if not quiet:
            print(f"  {label:<16} {status}")
            if not ok:
                print(f"    expected : {expected_val or '(not available)'}")
                print(f"    got      : {local_val}")

    if not quiet:
        print()
        if all_ok:
            print(_colour("✓ All checks passed.", True))
        else:
            print(_colour("✗ One or more checks FAILED. Do not use this file.", False))

    return all_ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="grid2op.check_release",
        description=(
            "Verify the integrity of a Grid2Op release asset (.whl or .tar.gz) "
            "by comparing its digests against those published on PyPI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a wheel with BLAKE2b-256 (default)
  grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl

  # Check with SHA-256 explicitly
  grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl --sha256

  # Check with all three algorithms
  grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl --all

  # Check specific algorithms
  grid2op.check_release Grid2Op-1.9.8.tar.gz --md5 --blake2b

  # Check without internet access using a local digest DB
  grid2op.check_release Grid2Op-1.9.8-py3-none-any.whl --offline release_digests.json
""",
    )

    parser.add_argument(
        "asset",
        type=Path,
        help="Path to the .whl or .tar.gz file to verify.",
    )

    algo_group = parser.add_argument_group("digest algorithms (default: --blake2b)")
    algo_group.add_argument(
        "--md5",
        action="store_true",
        help="Verify MD5 digest (weak, kept for legacy compatibility).",
    )
    algo_group.add_argument(
        "--sha256",
        action="store_true",
        help="Verify SHA-256 digest.",
    )
    algo_group.add_argument(
        "--blake2b",
        action="store_true",
        help="Verify BLAKE2b-256 digest.",
    )
    algo_group.add_argument(
        "--all",
        dest="all_algos",
        action="store_true",
        help="Verify all three digests (MD5 + SHA-256 + BLAKE2b-256).",
    )

    parser.add_argument(
        "--offline",
        metavar="DIGESTS_JSON",
        type=Path,
        default=None,
        help=(
            "Path to a local digest DB (JSON produced by fetch_release_digests.py). "
            "When omitted, digests are fetched live from PyPI. "
            "Use this for air-gapped environments or to avoid network calls."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output; rely solely on the exit code.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve requested algorithms
    if args.all_algos:
        algorithms = ["md5", "sha256", "blake2b_256"]
    else:
        algorithms = []
        if args.md5:
            algorithms.append("md5")
        if args.sha256:
            algorithms.append("sha256")
        if args.blake2b:
            algorithms.append("blake2b_256")
        if not algorithms:
            algorithms = ["blake2b_256"]   # default

    # Validate file path
    asset: Path = args.asset
    if not asset.exists():
        print(f"ERROR: File not found: {asset}", file=sys.stderr)
        return 1
    if not asset.is_file():
        print(f"ERROR: Not a file: {asset}", file=sys.stderr)
        return 1

    ok = verify(
        asset=asset,
        algorithms=algorithms,
        offline_db=args.offline,
        quiet=args.quiet,
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
