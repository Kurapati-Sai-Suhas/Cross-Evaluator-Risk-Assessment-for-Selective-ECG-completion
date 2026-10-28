"""Download the two PTB-XL v1.0.3 metadata CSVs (and PhysioNet's checksum list).

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.data.download_ptbxl_metadata

Only metadata is downloaded (no ECG waveforms). Files go to data/raw/ptbxl/.
Each CSV is verified against PhysioNet's official SHA256SUMS.txt. Files that
are already present and verified are not downloaded again (use --force to redo).

Exit codes: 0 = all files present and verified, 1 = checksum mismatch, 2 = download failed.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.data import ptbxl_io as io

FILES = ["ptbxl_database.csv", "scp_statements.csv"]
CHECKSUMS = "SHA256SUMS.txt"


def download(url: str, target: Path, timeout: float = 60.0) -> None:
    """Stream ``url`` to ``target`` via a temporary .part file (no half-written files)."""
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ptbxl-audit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
        total, done = response.length, 0
        while chunk := response.read(1 << 16):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {target.name}: {done / 1e6:6.2f} / {total / 1e6:.2f} MB", end="")
            else:
                print(f"\r  {target.name}: {done / 1e6:6.2f} MB", end="")
    print()
    part.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PTB-XL v1.0.3 metadata CSVs.")
    parser.add_argument("--config", default="configs/data_audit.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--force", action="store_true", help="Download again even if present")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    config = io.load_config(root / args.config)
    ds = config["dataset"]
    raw_dir = root / ds["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = ds["base_url"].rstrip("/") + "/"
    checksums_path = root / ds["checksums_file"]

    for name in [CHECKSUMS, *FILES]:
        target = raw_dir / name
        if target.is_file() and not args.force:
            if name == CHECKSUMS or io.verify_checksum(target, checksums_path)["status"] == "MATCH":
                print(f"  {name}: already present, skipping download")
                continue
        try:
            print(f"Downloading {base_url + name}")
            download(base_url + name, target)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"ERROR: could not download {name}: {exc}\n"
                f"Next action: check your internet connection and run again, or open\n"
                f"  {base_url + name}\n"
                f"in a browser, save the file as\n  {target}\nand re-run this script to verify it.",
                file=sys.stderr,
            )
            return 2

    exit_code = 0
    for name in FILES:
        result = io.verify_checksum(raw_dir / name, checksums_path)
        print(f"  {name}: sha256 {result['sha256'][:16]}...  official check: {result['status']}")
        if result["status"] != "MATCH":
            exit_code = 1
    if exit_code:
        print("ERROR: a checksum did not match. Delete the file and run again with --force.",
              file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
