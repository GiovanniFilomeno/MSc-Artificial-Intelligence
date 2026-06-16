# Privacy Audit

This repository is intended to be a public academic portfolio. The only personal name that should remain in portfolio-facing material is `Giovanni Filomeno`.

## Scope

The privacy pass searched for:

- student identifiers and old submission IDs;
- non-public filename variants of the owner's name;
- collaborator names found in old group artifacts;
- course or staff email addresses embedded in notebooks and generated artifacts;
- local absolute paths containing the machine username.

Public third-party datasets may still contain real-world names as data values. Those were not treated as collaborator or student identity leaks, because rewriting them would corrupt the assignments.

## Changes Applied

- Renamed notebooks, reports, archives, and submission folders to remove student IDs and non-public naming variants.
- Redacted identifiers inside Markdown, source files, notebooks, and legacy generated artifacts.
- Converted local absolute paths in Popper/Aleph and Computer Vision config files to relative paths.
- Removed generated caches, exported reports, ZIP bundles, model checkpoints, and local datasets that do not belong in a public portfolio repository.
- Strengthened `.gitignore` to block cache files, generated artifacts, archives, local datasets, and student-ID-like filenames.

## Verification

Run:

```bash
python3 scripts/privacy_scan.py
```

Expected result:

```text
Privacy scan passed. No configured identifiers found.
```

For a stricter local check, create an untracked `.privacy-patterns` file with one byte-regex per line. The latest pass also checked path names, text-visible content, and byte-level matches for the locally configured identifiers.
