#!/usr/bin/env python3
"""Scan the repository for common public-portfolio privacy leaks.

The script intentionally does not store private blocklist values in the
repository. For a stricter local pass, add one byte-regex per line to an
untracked `.privacy-patterns` file at the repository root.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PUBLIC_NAME = "Giovanni Filomeno"


def name_variant_patterns(name: str) -> list[tuple[str, re.Pattern[bytes]]]:
    parts = name.split()
    if len(parts) < 2:
        return []

    first, last = parts[0], parts[-1]
    variants = {
        f"{first}{last}",
        f"{first}_{last}",
        f"{first}-{last}",
        f"{last}_{first}",
        f"{last}-{first}",
    }
    return [
        ("non_public_owner_name_variant", re.compile(re.escape(variant).encode(), re.IGNORECASE))
        for variant in variants
    ]

PATTERNS: list[tuple[str, re.Pattern[bytes]]] = [
    ("student_id_like", re.compile(rb"(?i)(?<![A-Za-z0-9])(?:k|vk)\d{7,8}(?![A-Za-z0-9])")),
    ("email_address", re.compile(rb"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    ("mac_home_path", re.compile(rb"/Users/[^/\s\"]+")),
    ("windows_home_path", re.compile(rb"[A-Z]:(?:\\|\\\\)Users(?:\\|\\\\)[^\\\s\"]+", re.IGNORECASE)),
]
PATTERNS.extend(name_variant_patterns(ALLOWED_PUBLIC_NAME))

LOCAL_PATTERN_FILE = ROOT / ".privacy-patterns"

SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints"}
SKIP_EXTENSIONS = {
    ".avi",
    ".bmp",
    ".dylib",
    ".exe",
    ".gif",
    ".jpg",
    ".jpeg",
    ".mov",
    ".mp4",
    ".o",
    ".png",
    ".so",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts) or path.suffix.lower() in SKIP_EXTENSIONS


def load_local_patterns() -> list[tuple[str, re.Pattern[bytes]]]:
    if not LOCAL_PATTERN_FILE.exists():
        return []

    loaded: list[tuple[str, re.Pattern[bytes]]] = []
    for idx, line in enumerate(LOCAL_PATTERN_FILE.read_bytes().splitlines(), start=1):
        pattern = line.strip()
        if not pattern or pattern.startswith(b"#"):
            continue
        loaded.append((f"local_pattern_{idx}", re.compile(pattern, re.IGNORECASE)))
    return loaded


def main() -> int:
    hits: list[str] = []
    patterns = PATTERNS + load_local_patterns()
    scanner_path = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == scanner_path or should_skip(path):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue

        for label, pattern in patterns:
            if pattern.search(data):
                hits.append(f"{path.relative_to(ROOT)}\t{label}")
                break

    if hits:
        print("Privacy scan failed. Potential identifiers found:")
        print("\n".join(hits))
        return 1

    print("Privacy scan passed. No configured identifiers found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
