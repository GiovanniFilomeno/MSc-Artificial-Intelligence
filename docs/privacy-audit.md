# Privacy and Secret-Scanning Notes

This repository is intended for public portfolio review. The spaced public name `Giovanni Filomeno` may appear in portfolio-facing material; joined filename-style variants are treated as potential accidental identifiers by the local privacy scan.

## Automated scope

Run the local checks from the repository root:

```bash
python3 scripts/privacy_scan.py
python3 scripts/portfolio_quality.py secrets
```

`privacy_scan.py` checks file paths and readable file contents for:

- JKU-style student identifiers;
- email addresses;
- macOS and Windows home-directory paths;
- non-public filename-style variants of the owner's name;
- optional private patterns configured locally.

`portfolio_quality.py secrets` checks tracked files and any not-ignored local files for high-confidence private-key and credential formats. Notebook image, audio, and video payloads are excluded from this pattern check to avoid treating coincidental base64 substrings as credentials. Findings report only the path, location, and pattern type; matched values are never printed.

The current sanitized snapshot passes both checks.

## Private local patterns

For identifiers that should not be stored in the public repository, create an untracked `.privacy-patterns` file at the repository root with one case-insensitive byte-regex per line. The root `.gitignore` excludes this file.

Example structure, using placeholders rather than real identifiers:

```text
# One private byte-regex per line
PRIVATE_IDENTIFIER_PATTERN
ANOTHER_PRIVATE_PATTERN
```

Invalid local regular expressions fail the scan with their line number but do not echo the pattern.

## Important limitations

Passing these checks does not prove that the repository contains no private or restricted information. The checks do not fully inspect:

- images, audio, video, compiled binaries, serialized objects, or compressed archives;
- the contents of external Google Drive or other shared links;
- faces, voices, names, or identifiers visible in group-project media;
- commit-author metadata or copies retained outside the current reachable Git history;
- every provider-specific credential format.

Review those surfaces manually before promoting the repository. Use a GitHub-provided noreply address for public commits when a personal email address is not intended to be public. If a credential is ever committed, revoke or rotate it first; removing the text from a later commit is not sufficient.

## Public-history boundary

The current GitHub mirror began with a fresh single commit in June 2026. The privacy scan covers the current checkout, including ignored local configuration; the secret scan covers tracked and not-ignored local files. Neither check inspects Git object history. They cannot make guarantees about earlier private repositories, deleted public objects, external caches, local clones, screenshots, or forks.
