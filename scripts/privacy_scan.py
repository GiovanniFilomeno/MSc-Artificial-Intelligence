#!/usr/bin/env python3
"""Scan the repository for common public-portfolio privacy leaks.

The script intentionally does not store private blocklist values in the
repository. For a stricter local pass, add one byte-regex per line to an
untracked `.privacy-patterns` file at the repository root.

Findings identify their type and location without printing the matched value.
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


def is_in_skipped_directory(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def load_local_patterns() -> list[tuple[str, re.Pattern[bytes]]]:
    if not LOCAL_PATTERN_FILE.exists():
        return []

    loaded: list[tuple[str, re.Pattern[bytes]]] = []
    for idx, line in enumerate(LOCAL_PATTERN_FILE.read_bytes().splitlines(), start=1):
        pattern = line.strip()
        if not pattern or pattern.startswith(b"#"):
            continue
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(f"Invalid regex in .privacy-patterns at line {idx}; pattern not shown.")
            raise
        loaded.append((f"local_pattern_{idx}", compiled))
    return loaded


def redact_path(path: Path, patterns: list[tuple[str, re.Pattern[bytes]]]) -> str:
    """Return a path with any configured identifier replaced."""
    redacted = str(path).encode("utf-8", errors="replace")
    for _, pattern in patterns:
        redacted = pattern.sub(b"[REDACTED]", redacted)
    return redacted.decode("utf-8", errors="replace")


def main() -> int:
    hits: list[str] = []
    scan_errors: list[str] = []
    patterns = PATTERNS + load_local_patterns()
    scanner_path = Path(__file__).resolve()
    for path in sorted(ROOT.rglob("*")):
        try:
            resolved_path = path.resolve()
            resolved_path.relative_to(ROOT)
        except (OSError, ValueError):
            if path.is_symlink():
                relative_path = path.relative_to(ROOT)
                scan_errors.append(
                    f"{redact_path(relative_path, patterns)}\toutside_repository_symlink"
                )
            continue
        if (
            not path.is_file()
            or resolved_path == scanner_path
            or resolved_path == LOCAL_PATTERN_FILE.resolve()
            or is_in_skipped_directory(path)
        ):
            continue
        relative_path = path.relative_to(ROOT)

        for label, pattern in patterns:
            if pattern.search(str(relative_path).encode("utf-8", errors="replace")):
                hits.append(f"{redact_path(relative_path, patterns)}\tpath:{label}")
                break

        if path.suffix.lower() in SKIP_EXTENSIONS:
            continue

        try:
            data = path.read_bytes()
        except OSError:
            scan_errors.append(f"{redact_path(relative_path, patterns)}\tunreadable")
            continue

        for label, pattern in patterns:
            if pattern.search(data):
                hits.append(f"{redact_path(relative_path, patterns)}\tcontent:{label}")
                break

    if hits:
        print("Privacy scan failed. Potential identifiers found:")
        print("\n".join(hits))

    if scan_errors:
        print("Privacy scan could not inspect every repository path:")
        print("\n".join(scan_errors))

    if hits or scan_errors:
        return 1

    print("Privacy scan passed. No configured identifiers found.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except re.error:
        sys.exit(2)
