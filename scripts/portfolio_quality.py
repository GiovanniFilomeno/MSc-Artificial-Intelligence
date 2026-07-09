#!/usr/bin/env python3
"""Dependency-free public-portfolio quality checks.

The checks are intentionally static: they do not import coursework modules,
execute notebook cells, download data, or print values that resemble secrets.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import urllib.parse
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".avi",
    ".bmp",
    ".ckpt",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".ico",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".mov",
    ".mp4",
    ".o",
    ".obj",
    ".onnx",
    ".p",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".rdata",
    ".rar",
    ".so",
    ".tar",
    ".tgz",
    ".xls",
    ".xlsx",
    ".zip",
}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:(?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY|"
            r"PGP PRIVATE KEY BLOCK)-----"
        ),
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,255}\b")),
    ("github_fine_grained_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,255}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,255}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "openai_api_key",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9]{32,255}|"
            r"sk-(?:proj|svcacct)-[A-Za-z0-9_-]{20,255})\b"
        ),
    ),
    ("hugging_face_token", re.compile(r"\bhf_[A-Za-z0-9]{20,255}\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,255}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    (
        "pypi_token",
        re.compile(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{30,255}\b"),
    ),
    ("stripe_secret_key", re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{20,255}\b")),
    (
        "credential_in_url",
        re.compile(
            r"\b(?:https?|mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis)://"
            r"[^/@\s:]+:[^/@\s]+@",
            re.IGNORECASE,
        ),
    ),
)

GENERIC_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|"
    r"password|passwd)\b\s*[:=]\s*(['\"])([^'\"\s]{12,})\1",
    re.IGNORECASE,
)

PLACEHOLDER_TERMS = {
    "changeme",
    "dummy",
    "example",
    "password",
    "placeholder",
    "redacted",
    "replace_me",
    "secret",
    "token",
    "your_api_key",
    "your_password",
    "your_secret",
    "your_token",
}

MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)"
)
MARKDOWN_REFERENCE = re.compile(
    r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*(<[^>\n]+>|[^\s]+)",
    re.MULTILINE,
)


def is_repository_file(path: Path) -> bool:
    candidate = ROOT / path
    try:
        candidate.resolve().relative_to(ROOT)
    except (OSError, ValueError):
        return False
    return candidate.is_file()


def tracked_files() -> list[Path]:
    """Return tracked and not-ignored local repository files in stable order."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    candidates = (
        Path(raw.decode("utf-8", errors="surrogateescape"))
        for raw in result.stdout.split(b"\0")
        if raw
    )
    return sorted(path for path in candidates if is_repository_file(path))


def as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return "".join(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return ""


def notebook_surfaces(path: Path) -> Iterator[tuple[str, str]]:
    """Yield notebook text while excluding encoded binary MIME payloads."""
    with (ROOT / path).open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    if not isinstance(notebook, dict):
        raise ValueError("notebook root is not an object")

    metadata = notebook.get("metadata", {})
    yield "metadata", json.dumps(metadata, ensure_ascii=False)

    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("notebook cells is not a list")
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"notebook cell {cell_index} is not an object")
        yield f"cell {cell_index} source", as_text(cell.get("source", ""))
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise ValueError(f"notebook cell {cell_index} outputs is not a list")
        for output_index, output in enumerate(outputs):
            if not isinstance(output, dict):
                raise ValueError(
                    f"notebook cell {cell_index} output {output_index} is not an object"
                )
            output_type = output.get("output_type")
            location = f"cell {cell_index} output {output_index}"
            if output_type == "stream":
                yield location, as_text(output.get("text", ""))
            elif output_type == "error":
                error_text = "\n".join(
                    [
                        str(output.get("ename", "")),
                        str(output.get("evalue", "")),
                        as_text(output.get("traceback", [])),
                    ]
                )
                yield location, error_text
            else:
                data = output.get("data", {})
                if isinstance(data, dict):
                    for mime_type, value in data.items():
                        if mime_type.startswith("text/") or mime_type in {
                            "application/json",
                            "application/javascript",
                        }:
                            yield f"{location} {mime_type}", as_text(value)


def text_surfaces(path: Path) -> Iterator[tuple[str, str]]:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return
    if path.suffix.lower() == ".ipynb":
        yield from notebook_surfaces(path)
        return

    data = (ROOT / path).read_bytes()
    if b"\0" in data:
        return
    yield "content", data.decode("utf-8", errors="replace")


def is_placeholder(value: str) -> bool:
    if value.startswith(("${", "{{", "<")):
        return True
    normalized = value.strip().strip("<>[]{}$%").lower()
    return normalized in PLACEHOLDER_TERMS or normalized.startswith("your_")


def safe_display(value: str) -> str:
    """Redact any high-confidence secret shape from diagnostic text."""
    redacted = value
    for label, pattern in SECRET_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted


def check_secrets(paths: Iterable[Path]) -> int:
    findings: set[tuple[str, str, int, str]] = set()
    errors: list[str] = []

    for path in paths:
        try:
            surfaces = text_surfaces(path)
            for location, text in surfaces:
                for line_number, line in enumerate(text.splitlines() or [text], start=1):
                    for label, pattern in SECRET_PATTERNS:
                        if pattern.search(line):
                            findings.add((str(path), location, line_number, label))

                    for generic in GENERIC_SECRET_ASSIGNMENT.finditer(line):
                        if not is_placeholder(generic.group(2)):
                            findings.add(
                                (
                                    str(path),
                                    location,
                                    line_number,
                                    "assigned_secret_value",
                                )
                            )
        except (
            AttributeError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            errors.append(
                f"{safe_display(str(path))}: could not scan ({type(exc).__name__})"
            )

    if errors:
        print("Secret scan could not inspect every text surface:")
        for error in errors:
            print(f"- {error}")
    if findings:
        print("Potential secrets found; matched values are intentionally hidden:")
        for path, location, line_number, label in sorted(findings):
            print(
                f"- {safe_display(path)} | {location} line {line_number} | {label}"
            )
        return 1
    if errors:
        return 1

    print("Secret-shape scan passed. No high-confidence repository-text matches found.")
    return 0


def check_markdown_links(paths: Iterable[Path]) -> int:
    failures: list[tuple[str, int, str]] = []

    for path in paths:
        if path.suffix.lower() != ".md":
            continue
        content = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        matches = [
            *MARKDOWN_LINK.finditer(content),
            *MARKDOWN_REFERENCE.finditer(content),
        ]
        for match in sorted(matches, key=lambda item: item.start()):
            raw_target = match.group(1).strip("<>")
            if raw_target.startswith("//") or re.match(
                r"^(?:[a-z][a-z0-9+.-]*:|#)", raw_target, re.IGNORECASE
            ):
                continue

            target = urllib.parse.unquote(raw_target.split("#", 1)[0].split("?", 1)[0])
            if not target:
                continue
            line_number = content.count("\n", 0, match.start()) + 1
            destination = (ROOT / path.parent / target).resolve()
            try:
                destination.relative_to(ROOT)
                inside_repository = True
            except ValueError:
                inside_repository = False
            if inside_repository and destination.exists():
                continue

            failures.append((str(path), line_number, target))

    if failures:
        print("Broken local Markdown links:")
        for path, line_number, target in failures:
            print(
                f"- {safe_display(path)}:{line_number} -> {safe_display(target)}"
            )
        return 1

    print("Markdown link check passed.")
    return 0


def validate_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    shown_path = safe_display(str(path))
    try:
        with (ROOT / path).open(encoding="utf-8") as handle:
            notebook = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [
            f"{shown_path}: invalid notebook JSON "
            f"({type(exc).__name__})"
        ]

    if not isinstance(notebook, dict):
        return [f"{shown_path}: top-level notebook value is not an object"]
    if not isinstance(notebook.get("nbformat"), int):
        errors.append(f"{shown_path}: missing integer nbformat")
    if not isinstance(notebook.get("nbformat_minor"), int):
        errors.append(f"{shown_path}: missing integer nbformat_minor")
    if not isinstance(notebook.get("metadata", {}), dict):
        errors.append(f"{shown_path}: metadata is not an object")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        errors.append(f"{shown_path}: cells is not a list")
        return errors

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"{shown_path}: cell {index} is not an object")
            continue
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            errors.append(f"{shown_path}: cell {index} has an unknown cell_type")
        source = cell.get("source", [])
        if not isinstance(source, (str, list)):
            errors.append(
                f"{shown_path}: cell {index} source is neither text nor a list"
            )
        elif isinstance(source, list) and not all(
            isinstance(item, str) for item in source
        ):
            errors.append(f"{shown_path}: cell {index} source contains non-text items")
        if not isinstance(cell.get("metadata", {}), dict):
            errors.append(f"{shown_path}: cell {index} metadata is not an object")
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs", [])
            if not isinstance(outputs, list):
                errors.append(f"{shown_path}: code cell {index} outputs is not a list")
            elif not all(isinstance(output, dict) for output in outputs):
                errors.append(
                    f"{shown_path}: code cell {index} outputs contain non-objects"
                )
    return errors


def check_syntax(paths: Iterable[Path]) -> int:
    errors: list[str] = []
    python_count = 0
    notebook_count = 0
    warnings.simplefilter("ignore", SyntaxWarning)
    for path in paths:
        relative = str(path)
        if path.suffix.lower() == ".py":
            python_count += 1
            try:
                source = (ROOT / path).read_text(encoding="utf-8-sig")
                ast.parse(source, filename=relative)
            except SyntaxError as exc:
                errors.append(
                    f"{safe_display(str(path))}:{exc.lineno}: {exc.msg}"
                )
            except (OSError, UnicodeError) as exc:
                errors.append(
                    f"{safe_display(str(path))}: could not parse "
                    f"({type(exc).__name__})"
                )
        elif path.suffix.lower() == ".ipynb":
            notebook_count += 1
            errors.extend(validate_notebook(path))

    if errors:
        print("Static parse failures:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Static parse check passed: {python_count} Python files and "
        f"{notebook_count} notebooks."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "check",
        choices=("all", "links", "secrets", "syntax"),
        nargs="?",
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = tracked_files()
    checks = {
        "links": check_markdown_links,
        "secrets": check_secrets,
        "syntax": check_syntax,
    }

    if args.check != "all":
        return checks[args.check](paths)

    result = 0
    for name in ("secrets", "links", "syntax"):
        print(f"\n== {name} ==")
        result |= checks[name](paths)
    return result


if __name__ == "__main__":
    sys.exit(main())
